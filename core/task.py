"""core.task

Task object used by the SimPy environment.

Deadline-based version:
- No server failure model
- No backup server
- A task fails if its end-to-end response time exceeds its deadline
- Reads task params (size, demand, deadline) from data/task_parameters.xlsx
"""

import os
import pandas as pd

from config.params import params
from config.paths import DATA_DIR


class Task:
    def __init__(self, env, state, id, params_file: str = "task_parameters.xlsx"):
        self.env = env
        self.env_state = state
        self.id = id

        # Primary execution attributes (kept for compatibility with mainLoop logging)
        self.primaryNode = None
        self.primaryStarted = None
        self.primaryFinished = None
        self.primaryStat = None
        self.primary_service_time = None

        # Task parameters
        resolved = params_file
        if not os.path.isabs(resolved):
            resolved = os.path.join(DATA_DIR, resolved)

        task_info_df = pd.read_excel(resolved)
        task_row = task_info_df.loc[task_info_df["Task_ID"] == self.id]

        if task_row.empty:
            raise ValueError(f"Task_ID {self.id} not found in {resolved}")

        self.task_size = float(task_row["Task_Size"].values[0])
        self.computation_demand = float(task_row["Computation_Demand"].values[0])

        # Deadline (seconds)
        # Column name expected: "Deadline"
        self.deadline = float(task_row["Deadline"].values[0])

    # ---------------------------
    # Main execution entrypoint (called from mainLoop)
    # ---------------------------
    def execute_task(self, primary_server):
        self.primaryNode = primary_server
        self.primaryStarted = self.env.now
        yield self.env.process(self.primary())
        # primary() sets primaryFinished and primaryStat

    # ---------------------------
    # Primary execution (single server)
    # ---------------------------
    def primary(self):
        inpDelay, outDelay = self.calc_input_output_delay(self.primaryNode)

        # input delay (e.g., edge=0, cloud=task_size/bw)
        yield self.env.timeout(inpDelay)

        # queue + service
        with self.primaryNode.queue.request() as req:

            yield req
            self.env_state.assign_task_to_server(self.primaryNode.server_id, self)

            self.primary_service_time = self.computation_demand / self.primaryNode.processing_frequency
            yield self.env.timeout(self.primary_service_time)

        # output delay
        yield self.env.timeout(outDelay)

        # finish
        self.primaryFinished = self.env.now

        # Deadline check (end-to-end delay from start to finish)
        total_delay = self.primaryFinished - self.primaryStarted
        if total_delay <= self.deadline:
            self.primaryStat = "success"
        else:
            self.primaryStat = "failure"

        # bookkeeping
        self.env_state.complete_task(self.primaryNode.server_id, self, execute_time=self.primary_service_time)

    # ---------------------------
    # Network delay model
    # ---------------------------
    def calc_input_output_delay(self, server_object):
        if server_object.server_type == "Edge":
            inpDelay = 0.0
        else:
            inpDelay = float(self.task_size) / float(params.rsu_to_cloud_bandwidth)

        outDelay = inpDelay
        return inpDelay, outDelay
