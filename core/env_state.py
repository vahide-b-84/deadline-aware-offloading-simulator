"""core.env_state
Environment state container (deadline-based, no server failure).
State layout (exact):
  [ server1_freq, server1_load, server2_freq, server2_load, ..., serverN_freq, serverN_load,
    task_size, task_demand, task_deadline ]
=> size = 2*serverNo + 3
"""

import numpy as np
from config.params import params


class EnvironmentState:
    def __init__(self):
        # {server_id: {'server_object': obj, 'tasks_assigned': [...], 'load': float}}
        self.servers = {}
        # {task_id: task_object}
        self.tasks = {}
        self.num_completed_tasks = 0

    def add_server_and_init_environment(self, server_object):
        """Add a server object to the environment state."""
        server_id = int(server_object.server_id)
        self.servers[server_id] = {
            "server_object": server_object,
            "tasks_assigned": [],
            "load": 0.0,
        }

    def assign_task_to_server(self, server_id, task):
        """Assign a task object to a server (single primary selection)."""
        server_id = int(server_id)
        self.servers[server_id]["tasks_assigned"].append(task)
        self.servers[server_id]["load"] += float(task.computation_demand)

    def complete_task(self, server_id, task, execute_time=None):
        """
        Update environment bookkeeping when a task finishes.
        (execute_time kept optional for compatibility; not used here.)
        """
        server_id = int(server_id)
        tasks_assigned = self.servers[server_id]["tasks_assigned"]
        for i, t in enumerate(tasks_assigned):
            if t == task:
                self.servers[server_id]["load"] -= float(task.computation_demand)
                tasks_assigned.pop(i)
                self.num_completed_tasks += 1
                break

    def get_server_by_id(self, server_id):
        server_id = int(server_id)
        server_info = self.servers.get(server_id)
        return server_info["server_object"] if server_info else None

    def add_task(self, task_object):
        self.tasks[int(task_object.id)] = task_object

    def remove_task(self, task_id):
        self.tasks.pop(int(task_id), None)

    def get_task_by_id(self, task_id):
        return self.tasks.get(int(task_id))

    def reset(self):
        self.servers = {}
        self.tasks = {}
        self.num_completed_tasks = 0

    @staticmethod
    def normalize(val, min_val, max_val):
        return (val - min_val) / (max_val - min_val + 1e-8)

    def get_min_computation_demand(self):
        """
        Used by MainLoop "drain pending tasks" logic.
        We return a SMALL positive time-step that is likely enough for at least
        one task to complete soon.

        Heuristic:
          min_service_time ≈ min_demand / max_processing_frequency

        - Never returns None
        - Never returns 0 (returns at least 1e-3)
        """
        if not self.tasks:
            return 1e-3

        min_demand = min(float(t.computation_demand) for t in self.tasks.values())

        # Estimate max frequency among servers (fallback to 1.0)
        if self.servers:
            max_freq = max(float(info["server_object"].processing_frequency) for info in self.servers.values())
        else:
            max_freq = 1.0

        est = min_demand / (max_freq + 1e-8)
        return max(est, 1e-3)

    def get_state(self, task):
        """
        State vector (size = 2*serverNo + 3), in strict server order:
          [s1_freq, s1_load, s2_freq, s2_load, ... , sN_freq, sN_load, task_size, demand, deadline]
        """
        # IMPORTANT: keep a stable order by server_id
        server_ids_sorted = sorted(self.servers.keys())

        # If frequency ranges exist, use them for normalization; otherwise infer from current servers
        freqs = [float(self.servers[sid]["server_object"].processing_frequency) for sid in server_ids_sorted]
        if getattr(params, "EDGE_PROCESSING_FREQ_RANGE", None) and getattr(params, "CLOUD_PROCESSING_FREQ_RANGE", None):
            freq_min = float(params.EDGE_PROCESSING_FREQ_RANGE[0])
            freq_max = float(params.CLOUD_PROCESSING_FREQ_RANGE[1])
        else:
            freq_min = min(freqs) if freqs else 0.0
            freq_max = max(freqs) if freqs else 1.0

        # Build per-server features sequentially
        state_list = []
        loads = [float(self.servers[sid]["load"]) for sid in server_ids_sorted]
        max_local_load = max(loads) if loads else 1.0

        for sid in server_ids_sorted:
            s = self.servers[sid]["server_object"]
            f = float(s.processing_frequency)
            l = float(self.servers[sid]["load"])

            norm_f = float(self.normalize(f, freq_min, freq_max))
            norm_l = float(l / (max_local_load + 1e-8)) if max_local_load > 0 else 0.0

            state_list.extend([norm_f, norm_l])

        # Task features (always appended at the end)
        norm_task_size = float(self.normalize(float(task.task_size), params.TASK_SIZE_RANGE[0], params.TASK_SIZE_RANGE[1]))
        norm_demand = float(self.normalize(float(task.computation_demand), params.Low_demand, params.High_demand))
        norm_deadline = float(self.normalize(float(task.deadline), params.task_deadline[0], params.task_deadline[1]))

        state_list.extend([norm_task_size, norm_demand, norm_deadline])

        return np.array(state_list, dtype=np.float32)
