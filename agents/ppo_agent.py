# PPO_template.py
# Project-specific PPO implementation (on-policy), designed to work with the existing simulation codebase.
# Notes:
# - Rollout buffer is collected within an episode.
# - train_step() is intended to run at the end of the episode.
# - Function signatures keep compatibility with the DQN interface used elsewhere in the project.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical


class PPOPolicyNetwork(nn.Module):
    # Actor network: maps state -> action logits
    def __init__(self, input_dim, output_dim, hidden_layers, activation="tanh"):
        super(PPOPolicyNetwork, self).__init__()
        layers = []
        prev_dim = input_dim

        # Build MLP body
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))

            # Activation selection
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unsupported activation function: {activation}")

            prev_dim = h

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        # Returns action logits (Categorical distribution will be formed from logits)
        x = self.hidden_layers(x)
        return self.output_layer(x)  # logits


class PPOValueNetwork(nn.Module):
    # Critic network: maps state -> scalar value V(s)
    def __init__(self, input_dim, hidden_layers, activation="tanh"):
        super(PPOValueNetwork, self).__init__()
        layers = []
        prev_dim = input_dim

        # Build MLP body
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))

            # Activation selection
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unsupported activation function: {activation}")

            prev_dim = h

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, 1)

    def forward(self, x):
        # Squeeze last dimension to return shape [batch] instead of [batch, 1]
        x = self.hidden_layers(x)
        return self.output_layer(x).squeeze(-1)  # V(s)


class PPOAgent:
    """
    Project-specific PPO agent (on-policy).

    Key assumptions in this project:
    - Rollout data is collected only for the current episode.
    - train_step() is called at the end of the episode.
    """

    def __init__(
        self,
        num_states,
        num_actions,
        hidden_layers,
        device="cpu",
        gamma=0.99,
        actor_lr=3e-4,
        critic_lr=1e-3,
        clip_eps=0.2,
        k_epochs=2,
        batch_size=64,
        entropy_coef=0.01,
        reward_scale=1.0,
        gae_lambda=0.95,        # Note: this implementation uses TD(0)-style advantage (not full GAE rollout)
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        activation="tanh",
        min_rollout=8,          # Minimum number of transitions required before performing an update
    ):
        self.device = torch.device(device)
        self.num_actions = num_actions

        # PPO / RL hyperparameters
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.k_epochs = k_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.reward_scale = reward_scale
        self.gae_lambda = gae_lambda
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.min_rollout = int(min_rollout)

        # Policy networks:
        # - policy_net: trainable policy
        # - policy_old: frozen snapshot used for sampling and stable old log-prob computation
        self.policy_net = PPOPolicyNetwork(
            num_states, num_actions, hidden_layers, activation
        ).to(self.device)

        self.policy_old = PPOPolicyNetwork(
            num_states, num_actions, hidden_layers, activation
        ).to(self.device)
        self.policy_old.load_state_dict(self.policy_net.state_dict())

        # Value network (critic)
        self.value_net = PPOValueNetwork(
            num_states, hidden_layers, activation
        ).to(self.device)

        # Separate optimizers for actor and critic
        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=actor_lr)
        self.optimizer_value = optim.Adam(self.value_net.parameters(), lr=critic_lr)

        # Episode rollout buffer (cleared after train_step)
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.old_log_probs = []

        # Compatibility placeholder:
        # Some parts of the project check agent.replay_buffer length (DQN-style).
        self.replay_buffer = []

    # -----------------------------
    # utilities
    # -----------------------------
    def _to_tensor(self, x):
        # Convert numpy-like input to float32 tensor on target device
        return torch.tensor(np.array(x), dtype=torch.float32, device=self.device)

    # -----------------------------
    # action selection
    # -----------------------------
    def select_action(self, state, epsilon, use_softmax=False, temperature=1.5):
        """
        PPO action selection.
        NOTE: The signature matches other agents in this project for compatibility.
              epsilon/use_softmax/temperature are not used by PPO here.
        """
        state_tensor = self._to_tensor(state).unsqueeze(0)
        with torch.no_grad():
            # Sample from policy_old for more stable on-policy behavior
            logits = self.policy_old(state_tensor)
            if not torch.isfinite(logits).all():
                # Fallback: if logits become invalid, return a random action
                return int(np.random.randint(0, self.num_actions))
            dist = Categorical(logits=logits)
            action = dist.sample()
        return int(action.item())

    # -----------------------------
    # store transition
    # -----------------------------
    def store_transition(self, s, a, r, s_next, done=False):
        # Ignore unresolved transitions (reward is required)
        if r is None:
            return

        # Store rollout transition for the current episode
        self.states.append(np.array(s, copy=False))
        self.actions.append(int(a))
        self.rewards.append(float(r))
        self.next_states.append(np.array(s_next, copy=False))
        self.dones.append(bool(done))

        # Store old log-prob using policy_old (standard PPO approach)
        with torch.no_grad():
            s_tensor = self._to_tensor(s).unsqueeze(0)
            logits = self.policy_old(s_tensor)
            if not torch.isfinite(logits).all():
                self.old_log_probs.append(0.0)
            else:
                dist = Categorical(logits=logits)
                a_tensor = torch.tensor(int(a), dtype=torch.int64, device=self.device)
                log_prob = dist.log_prob(a_tensor).item()
                if not np.isfinite(log_prob):
                    log_prob = 0.0
                self.old_log_probs.append(float(log_prob))

    # -----------------------------
    # training: end of episode
    # -----------------------------
    def train_step(self):
        # Perform PPO update using the collected episode rollout buffer
        N = len(self.states)
        if N == 0:
            return

        # If too few samples, skip update and clear rollout
        # (important for local RSUs that might collect only a couple of transitions)
        if N < self.min_rollout:
            self.clear_rollout()
            return

        # If no terminal flags were recorded, mark the last transition as terminal
        if not any(self.dones):
            self.dones[-1] = True

        # Convert rollout data to tensors
        states = self._to_tensor(self.states)
        next_states = self._to_tensor(self.next_states)
        actions = torch.tensor(self.actions, dtype=torch.int64, device=self.device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=self.device) * float(self.reward_scale)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor(self.old_log_probs, dtype=torch.float32, device=self.device)

        # Basic numeric sanity checks (avoid propagating NaNs/Infs)
        if (not torch.isfinite(states).all()) or (not torch.isfinite(next_states).all()) or (not torch.isfinite(rewards).all()):
            self.clear_rollout()
            return

        # Current value estimates V(s)
        values = self.value_net(states)

        with torch.no_grad():
            # TD targets: r + gamma * V(s') * (1-done)
            next_values = self.value_net(next_states)
            td_targets = rewards + self.gamma * next_values * (1.0 - dones)

            # Advantage estimate (TD error): A = td_target - V(s)
            advantages = td_targets - values

            # Normalize advantages safely:
            # use unbiased=False to avoid NaN when N is small
            adv_mean = advantages.mean()
            adv_std = advantages.std(unbiased=False)

            # If std is too small or invalid, fallback to mean-centering only
            if (not torch.isfinite(adv_std)) or adv_std.item() < 1e-8:
                advantages = advantages - adv_mean
            else:
                advantages = (advantages - adv_mean) / (adv_std + 1e-8)

            # If still invalid, skip update
            if not torch.isfinite(advantages).all():
                return

        # Safe batch size for small rollouts
        batch_size = min(int(self.batch_size), N)

        # PPO optimization epochs
        for _ in range(int(self.k_epochs)):
            idx = np.arange(N)
            np.random.shuffle(idx)

            for start in range(0, N, batch_size):
                end = start + batch_size
                batch_idx = idx[start:end]

                # Minibatch tensors
                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_advantages = advantages[batch_idx]
                b_td_targets = td_targets[batch_idx].detach()
                b_old_log_probs = old_log_probs[batch_idx]

                # New policy logits for minibatch
                logits = self.policy_net(b_states)
                if not torch.isfinite(logits).all():
                    # Skip minibatch if logits are invalid
                    continue

                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                # PPO ratio: exp(logp_new - logp_old)
                log_ratio = log_probs - b_old_log_probs
                log_ratio = torch.clamp(log_ratio, -20.0, 20.0)
                ratios = torch.exp(log_ratio)

                # Clipped surrogate objective
                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                # Value loss (critic)
                new_values = self.value_net(b_states)
                value_loss = self.value_loss_coef * nn.MSELoss()(new_values, b_td_targets)

                # Total loss (actor + critic)
                total_loss = policy_loss + value_loss

                if not torch.isfinite(total_loss):
                    continue

                # Backprop
                self.optimizer_policy.zero_grad(set_to_none=True)
                self.optimizer_value.zero_grad(set_to_none=True)
                total_loss.backward()

                # Gradient clipping for stability
                nn.utils.clip_grad_norm_(
                    list(self.policy_net.parameters()) + list(self.value_net.parameters()),
                    self.max_grad_norm,
                )

                # Optimizer steps
                self.optimizer_policy.step()
                self.optimizer_value.step()

        # After update: sync policy_old with new policy
        self.policy_old.load_state_dict(self.policy_net.state_dict())

        # Clear episode rollout buffer
        self.clear_rollout()

    # -----------------------------
    # saving / loading
    # -----------------------------
    def save_model(self, path):
        # Save actor/critic parameters and optimizer states
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "policy_old": self.policy_old.state_dict(),
                "value_net": self.value_net.state_dict(),
                "opt_policy": self.optimizer_policy.state_dict(),
                "opt_value": self.optimizer_value.state_dict(),
            },
            path,
        )

    def load_model(self, path):
        # Load actor/critic parameters and (optionally) optimizer states
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        if "policy_old" in checkpoint:
            self.policy_old.load_state_dict(checkpoint["policy_old"])
        else:
            self.policy_old.load_state_dict(checkpoint["policy_net"])
        self.value_net.load_state_dict(checkpoint["value_net"])
        if "opt_policy" in checkpoint:
            self.optimizer_policy.load_state_dict(checkpoint["opt_policy"])
        if "opt_value" in checkpoint:
            self.optimizer_value.load_state_dict(checkpoint["opt_value"])

    # -----------------------------
    # rollout management
    # -----------------------------
    def clear_rollout(self):
        # Clear per-episode rollout buffer
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.old_log_probs.clear()
