# Project_main.py
# ------------------------------------------------------------
# Entry point for the simulator (deadline-based version).
# - No FAILURE_STATE / Alpha / failure-rate logic.
# ------------------------------------------------------------

import os
import sys

# Ensure the project root is on PYTHONPATH when running directly.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from io_utils.save_parameters_and_logs import save_params_and_logs
from config.params import params
from core.main_loop import MainLoop


def build_model():
    """Build and return the model/agent object based on params.model_summary.

    Contract:
      - DQN/PPO: select_action(state, epsilon) -> int
    """
    model_name = str(params.model_summary).strip().lower()

    if model_name == "dqn":
        from agents.dqn_agent import DQNAgent

        model = DQNAgent(
            num_states=params.num_states,
            num_actions=params.num_actions,
            hidden_layers=params.hidden_layers_dqn,
            device="cpu",
            gamma=params.gamma_dqn,
            lr=params.lr_dqn,
            tau=params.tau_dqn,
            buffer_size=params.buffer_capacity_dqn,
            batch_size=params.batch_size_dqn,
            activation=params.af_dqn,
        )
        print("DQNAgent is set.")
        return model

    if model_name == "ppo":
        from agents.ppo_agent import PPOAgent

        model = PPOAgent(
            num_states=params.num_states,
            num_actions=params.num_actions,
            hidden_layers=params.hidden_layers_ppo,
            device="cpu",
            gamma=params.gamma_ppo,
            actor_lr=params.actor_lr_ppo,
            critic_lr=params.critic_lr_ppo,
            clip_eps=params.clip_eps_ppo,
            k_epochs=params.k_epochs_ppo,
            batch_size=params.batch_size_ppo,
            entropy_coef=params.entropy_coef_ppo,
            reward_scale=params.reward_scale_ppo,
            gae_lambda=params.gae_lambda_ppo,
            value_loss_coef=params.value_loss_coef_ppo,
            max_grad_norm=params.max_grad_norm_ppo,
            activation=params.af_ppo,
        )
        print("PPOAgent is set.")
        return model

    raise ValueError(
        f"Unsupported params.model_summary='{params.model_summary}'. "
        "Use one of: 'dqn', 'ppo'."
    )


def run_simulation():

    model = build_model()
    ml = MainLoop(model, params.total_episodes, params.taskno, params.num_states, params.num_actions)
    ml.EP()
    # deadline-based saver (no failure_state)
    save_params_and_logs(params, ml.log_data, ml.task_Assignments_info)

def main():
    scenario_type = getattr(params, "SCENARIO_TYPE", "heterogeneous")
    print(f"##### Running simulation for scenario: {scenario_type} #####")
    run_simulation()


if __name__ == "__main__":
    main()
