# Deadline-Aware DRL Offloading Simulator (Single-Action)

A simplified, deadline-focused simulator for **edge–cloud task offloading** using **Deep Reinforcement Learning (DRL)**.

This repository is a **simplified** version of an earlier “reliable offloading” project:
- ✅ **Single decision (single action) per task:** choose **one primary server** (Edge/Cloud)
- ✅ **Deadline-aware outcomes:** each task is **success** or **deadline miss** based on its completion time vs. deadline
- ❌ **No failure model**
- ❌ **No fault-tolerance / recovery strategies** (no Retry / Recovery Block / First Result)
- ❌ **No backup server selection**

Supported DRL models:
- **DQN**
- **PPO**

---

## Project overview

### What the simulator does
For each incoming task:
1. Builds a **state vector** from the current environment (queues/load + task features).
2. The agent selects **one discrete action**: a server index in `[0 .. num_actions-1]`.
3. The task is executed on the selected server.
4. Reward is computed from **delay** and **deadline satisfaction**.

### Action space (important)
Actions are **0-based**:
- `action = 0` → `Server_ID = 1`
- `action = 1` → `Server_ID = 2`
- ...
- `action = serverNo-1` → `Server_ID = serverNo`

---

## Repository structure (high level)

```
project_root/
├── Project_main.py                 # entry point
├── config/
│   ├── params.py                   # experiment + RL hyperparameters
│   └── paths.py                    # DATA_DIR, RESULT_DIR, etc.
├── core/
│   ├── main_loop.py                # episode loop + action selection + training calls
│   ├── env_state.py                # environment state builder
│   ├── server.py                   # server model
│   └── task.py                     # task execution + deadline success/failure
├── agents/
│   ├── dqn_agent.py                # DQN implementation
│   └── ppo_agent.py                # PPO implementation
├── io_utils/
│   ├── save_parameters_and_logs.py # writes Excel logs + charts
│   └── post_process_results.py     # optional: merge/enrich result Excels
├── tools/
│   └── generate_server_and_task_parameters.py  # generates server/task Excel inputs
└── data/
   ├── heterogeneous_server_info.xlsx
   ├── homogeneous_server_info.xlsx
   └── task_parameters.xlsx
```

---

## Installation

### 1) Create and activate a Python environment
Conda example:
```bash
conda create -n myenv_drl python=3.10 -y
conda activate myenv_drl
```

### 2) Install dependencies
If you already have a `requirements.txt`, use it:
```bash
pip install -r requirements.txt
```

Otherwise, the typical minimum set is:
```bash
pip install numpy pandas simpy openpyxl torch
```

> Notes:
> - `torch` CPU is sufficient.
> - `openpyxl` is used to generate **Excel-native charts**.

---

## Quick start

### Step A — Generate input Excel files
Run the generator (creates/updates files inside `data/`):
```bash
python tools/generate_server_and_task_parameters.py
```

This should produce:
- `data/heterogeneous_server_info.xlsx`
- `data/homogeneous_server_info.xlsx`
- `data/task_parameters.xlsx` (includes deadlines)

### Step B — Run a training session
Edit `config/params.py`:
- `model_summary = "dqn"` or `"ppo"`
- `SCENARIO_TYPE = "heterogeneous"` or `"homogeneous"`
- Set `total_episodes`, `taskno`, etc.

Then run:
```bash
python Project_main.py
```

### Step C — (Optional) Post-process results
If you want extra aggregation/enrichment over generated result Excel files:
```bash
python io_utils/post_process_results.py
```

---

## Outputs

During/after training, the simulator writes an Excel report per run including:
- **Per-episode logs** (reward/delay)
- **TaskAssignments**: per-task selected server, start/finish time, deadline, and status
- **Summary metrics** such as:
  - On-time rate
  - Deadline miss rate
  - Rolling averages (e.g., last 40 episodes)
- Excel-native charts (no PNG export)

The output location and file naming are controlled in:
- `config/paths.py`
- `io_utils/save_parameters_and_logs.py`

---

## Configuration notes

### Numbering consistency
Make sure these are consistent:
- `params.NUM_EDGE_SERVERS`
- `params.NUM_CLOUD_SERVERS`
- `params.serverNo = NUM_EDGE_SERVERS + NUM_CLOUD_SERVERS`
- `params.num_actions = params.serverNo`

And the server Excel must contain:
- `Server_ID` = `1..serverNo` (contiguous)

### Deadline logic
Deadline success/failure is determined in:
- `core/task.py` via `task.deadline` and the task completion time.

Reward shaping is handled in:
- `core/main_loop.py` (typically uses delay and applies a penalty on deadline miss)

---

## Troubleshooting

### Cloud server is never selected
Almost always caused by **action ↔ server_id mapping** issues.
This project expects:
- **actions are 0-based**
- `server_id = action + 1`


### Excel read errors
Ensure the server Excel has a sheet named:
- `Servers`

and columns:
- `Server_ID`, `Server_Type`, `Processing_Frequency`

---

## License
This project is licensed under the MIT License.
See the LICENSE file for details.