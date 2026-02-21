# configuration.py
# Central configuration for the simulation + RL agents (deadline-based, no server failure).

class parameters:
    # ======================================================================
    # Experiment settings
    # ======================================================================
    SCENARIO_TYPE = "heterogeneous"  # Options: "homogeneous", "heterogeneous"
    model_summary = "dqn"            # Options: "dqn", "ppo"
    total_episodes = 5               # e.g., 100

    # ======================================================================
    # Infrastructure: servers
    # ======================================================================
    NUM_EDGE_SERVERS = 6
    NUM_CLOUD_SERVERS = 2
    serverNo = NUM_EDGE_SERVERS + NUM_CLOUD_SERVERS

    # ======================================================================
    # Workload: tasks
    # ======================================================================
    TASK_ARRIVAL_RATE = 0.5         # Task inter-arrival time (or rate depending on your simulator)
    TASK_SIZE_RANGE = (10, 100)
    Low_demand, High_demand = 1, 100
    taskno = 200

    # Task deadline range (seconds) - used in preprocess to generate a per-task deadline column
    task_deadline = (0.5, 25)

    # ======================================================================
    # Network model
    # ======================================================================
    rsu_to_cloud_bandwidth = 8       # Mb/s

    # ======================================================================
    # Infrastructure
    # ======================================================================

    EDGE_PROCESSING_FREQ_RANGE = (10, 15)
    CLOUD_PROCESSING_FREQ_RANGE = (30, 60)


    # ======================================================================
    # RL hyperparameters
    # ======================================================================

    # ------------- DQN --------------
    hidden_layers_dqn = [128, 64]
    af_dqn = "relu"
    lr_dqn = 5e-4
    gamma_dqn = 0.90
    tau_dqn = 0.005
    buffer_capacity_dqn = 200_000
    batch_size_dqn = 256
    epsilon_start_dqn = 1.0
    epsilon_end_dqn = 0.01
    epsilon_decay_dqn = 300

    # ----------- PPO --------------
    hidden_layers_ppo = [64, 32]
    af_ppo = "tanh"
    actor_lr_ppo = 1e-4
    critic_lr_ppo = 5e-4
    gamma_ppo = 0.90
    clip_eps_ppo = 0.2
    k_epochs_ppo = 2
    batch_size_ppo = 64
    entropy_coef_ppo = 0.01
    reward_scale_ppo = 1
    gae_lambda_ppo = 0.95
    value_loss_coef_ppo = 0.5
    max_grad_norm_ppo = 0.5
