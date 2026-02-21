# mainLoop.py  (multi-model: DQN / PPO) - DEADLINE-BASED, SINGLE SERVER ACTION
# - No failure model, no backup server, no z parameter
# - Action = choose primary server 
# - Reward uses deadline success/fail from task.primaryStat and delay

from core.server import Server
from core.task import Task
from core.env_state import EnvironmentState
from config.params import params
from config.paths import DATA_DIR

import simpy
import numpy as np
import os
import pandas as pd
import math


class MainLoop:
    def __init__(self, model, total_episodes, maxtaskno, num_states, num_actions):
        self.model = model
        self.num_states = num_states
        self.num_actions = num_actions
        self.total_episodes = total_episodes
        self.model_name = str(getattr(params, "model_summary", "dqn")).strip().lower()
        self.rewardsAll = []
        self.ep_reward_list = []
        self.ep_delay_list = []
        self.avg_reward_list = []
        self.this_episode = 0

        self.G_state = []
        self.G_action = None  # DQN/PPO: int action index

        self.episodic_reward = 0
        self.episodic_delay = 0

        # tempbuffer[taskCounter] = (s, a, r, s')
        self.tempbuffer = {}
        self.taskCounter = 1
        self.pendingList = []
        self.maxTask = maxtaskno

        self.env = None
        self.env_state = None
        self.log_data = []
        self.task_Assignments_info = []

        self.SCENARIO_TYPE = getattr(params, "SCENARIO_TYPE", "heterogeneous")

    # ---------------------------
    # EPISODE LOOP
    # ---------------------------
    def EP(self):
        while self.this_episode < self.total_episodes:
            self.this_episode += 1
            self.episodic_reward = 0
            self.episodic_delay = 0
            self.tempbuffer = {}
            self.taskCounter = 1
            self.pendingList = []

            self.env = simpy.Environment()
            self.env_state = EnvironmentState()
            self.env_state.reset()

            self.setServers()
            self.env.process(self.Iteration())
            self.env.run()

    # ---------------------------
    # epsilon schedule (DQN only; PPO ignores epsilon in its select_action signature)
    # ---------------------------
    def get_epsilon(self, episode):
        if self.model_name != "dqn":
            return 0.0
        eps_start = getattr(params, "epsilon_start_dqn", 1.0)
        eps_end = getattr(params, "epsilon_end_dqn", 0.01)
        eps_decay = getattr(params, "epsilon_decay_dqn", 300)
        return max(eps_end, eps_start - (episode / float(eps_decay)))

    # ---------------------------
    # MAIN SIMULATION ITERATION
    # ---------------------------
    def Iteration(self):
        while self.taskCounter <= self.maxTask:
            yield self.env.timeout(np.random.poisson(1 / params.TASK_ARRIVAL_RATE))

            task = Task(self.env, self.env_state, self.taskCounter)
            self.env_state.add_task(task)

            # build state vector
            self.G_state = self.env_state.get_state(task)

            # complete s' for previous transition and train on any resolved tasks
            if self.taskCounter > 1:
                prev = list(self.tempbuffer[self.taskCounter - 1])
                prev[3] = self.G_state
                self.tempbuffer[self.taskCounter - 1] = tuple(prev)
                self.add_train()
            
            eps = self.get_epsilon(self.this_episode)
            action_index = self.model.select_action(self.G_state, eps)
            self.G_action = int(action_index)
            primary_server = self.extract_primary_from_index(self.G_action)
            
            # store pending transition (reward filled later)
            self.tempbuffer[self.taskCounter] = (self.G_state, self.G_action, None, [])
            self.env.process(task.execute_task(primary_server))
            self.pendingList.append(self.taskCounter)

            self.taskCounter += 1

        # finalize last transition next-state placeholder
        if self.taskCounter > 1:
            last = list(self.tempbuffer[self.taskCounter - 1])
            last[3] = self.G_state
            self.tempbuffer[self.taskCounter - 1] = tuple(last)

        # drain pending tasks until all rewards resolved
        while len(self.pendingList) > 0:
            yield_time = self.env_state.get_min_computation_demand()
            yield self.env.timeout(yield_time)
            self.add_train()

        # PPO: update policy at end of episode (on-policy)
        if self.model_name == "ppo":
            self.model.train_step()

        # episode logs
        self.ep_reward_list.append(self.episodic_reward)
        self.ep_delay_list.append(self.episodic_delay)

        avg_reward = np.mean(self.ep_reward_list[-40:])
        avg_delay = np.mean(self.ep_delay_list[-40:])
        self.log_data.append((self.this_episode, avg_reward, self.episodic_reward, avg_delay))
        self.avg_reward_list.append(avg_reward)
        print(f"Episode {self.this_episode} | Avg Reward: {avg_reward:.3f} | This Episode: {self.episodic_reward:.3f}")

    # ---------------------------
    # REWARD CALCULATION (deadline-based)
    # Uses: task.primaryStarted, task.primaryFinished, task.primaryStat
    # task.primaryStat should be "success"/"failure" based on deadline in task.py
    # ---------------------------
    def calcReward(self, taskID):
        task = self.env_state.get_task_by_id(taskID)

        primaryStat = task.primaryStat
        primaryFinished = task.primaryFinished
        primaryStarted = task.primaryStarted

        # not finished yet
        if primaryStarted is None or primaryFinished is None or primaryStat is None:
            return None, None

        delay = primaryFinished - primaryStarted

        if primaryStat == "failure":
            # deadline miss penalty
            failure_penalty_weight = 3.0
            reward = -failure_penalty_weight * delay
            if reward > -3:
                reward = -3
        else:
            # success reward (same shaping as before)
            reward = (math.log(1 - (1 / math.exp(math.sqrt(delay)))) / math.log(0.995))

        return reward, delay
    # ---------------------------
    # TRAINING (multi-model)
    # ---------------------------
    def add_train(self):
        removeList = []

        for task_counter in list(self.pendingList):
            reward, delay = self.calcReward(task_counter)
            if reward is None:
                continue

            self.episodic_reward += reward
            self.episodic_delay += delay
            self.rewardsAll.append(reward)

            temp = list(self.tempbuffer[task_counter])
            temp[2] = reward
            self.tempbuffer[task_counter] = tuple(temp)

            s, a, r, s_ = self.tempbuffer[task_counter]

            if self.model_name == "dqn":
                self.model.store_transition((s, int(a), r, s_))
                self.model.train_step()

            elif self.model_name == "ppo":
                self.model.store_transition(s, int(a), r, s_, done=False)

            removeList.append(task_counter)

        for t in removeList:
            self.pendingList.remove(t)
            task = self.env_state.get_task_by_id(t)

            # simplified assignment log (no backup, no z)
            self.task_Assignments_info.append(
                (
                    self.this_episode,
                    task.id,
                    task.primaryNode.server_id if task.primaryNode is not None else None,
                    task.primaryStarted,
                    task.primaryFinished,
                    task.primaryStat,
                    getattr(task, "deadline", None),
                )
            )
            self.env_state.remove_task(t)

    # ---------------------------
    # SERVERS
    # ---------------------------
    def setServers(self):
        # You can keep SCENARIO_TYPE if it still decides which server excel to use
        excel_file = "homogeneous_server_info.xlsx" if self.SCENARIO_TYPE == "homogeneous" else "heterogeneous_server_info.xlsx"
        excel_file = os.path.join(DATA_DIR, excel_file)

        # NEW: single sheet (no FAILURE_STATE). Prefer "Servers".
        # If your excel still uses another sheet name, change it here once.
        sheet_name = "Servers"

        server_info_df = pd.read_excel(excel_file, sheet_name=sheet_name)

        for _, row in server_info_df.iterrows():
            server_id = int(row["Server_ID"])
            server_type = str(row["Server_Type"])
            processing_frequency = float(row["Processing_Frequency"])

            # failure_rate removed -> Server signature must be updated accordingly
            server = Server(self.env, server_type, server_id, processing_frequency)
            self.env_state.add_server_and_init_environment(server)

    # ---------------------------
    # ACTION DECODING (single primary)
    # ---------------------------
    def extract_primary_from_index(self, action_index: int):
        idx = int(action_index)

        # action must be 0-based in [0..serverNo-1]
        if idx < 0 or idx >= params.serverNo:
            raise ValueError(f"Action index out of range: {idx}, serverNo={params.serverNo}")

        server_id = idx + 1  # 0-based action -> 1-based server_id

        primary_server = self.env_state.get_server_by_id(server_id)
        if primary_server is None:
            raise ValueError(
                f"Invalid mapping -> server_id={server_id}. "
                f"Loaded servers: {sorted(self.env_state.servers.keys())}"
            )
        return primary_server

