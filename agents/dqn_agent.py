# DQN_template.py
# Deep Q-Network (DQN) implementation used in this project.
# Includes:
#   - A configurable MLP Q-network (DQNNetwork)
#   - A DQNAgent with:
#       * epsilon-greedy exploration
#       * optional softmax action sampling (Boltzmann exploration)
#       * replay buffer
#       * target network with soft update (Polyak averaging)

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

class DQNNetwork(nn.Module):
    """
    Simple feed-forward MLP for approximating Q(s, a).
    Architecture is controlled by:
      - input_dim: state dimension
      - output_dim: number of discrete actions
      - hidden_layers: list of hidden layer widths
      - activation: activation function name
    """
    def __init__(self, input_dim, output_dim, hidden_layers, activation='relu'):
        super(DQNNetwork, self).__init__()
        layers = []
        prev_dim = input_dim

        # Build MLP hidden layers
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))

            # Activation selection (must match allowed strings)
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unsupported activation function: {activation}")

            prev_dim = h

        self.hidden_layers = nn.Sequential(*layers)

        # Final linear layer outputs Q-values for all actions
        self.output_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        """Forward pass: returns raw Q-values (no softmax)."""
        x = self.hidden_layers(x)
        return self.output_layer(x)

class DQNAgent:
    """
    DQN Agent with:
      - policy_net: online Q-network
      - target_net: target Q-network (lagging copy)
      - replay_buffer: list of (s, a, r, s') transitions
      - soft target updates using tau
    """
    def __init__(self, num_states, num_actions, hidden_layers, device='cpu', gamma=0.99, lr=1e-3, tau=0.005, buffer_size=100000, batch_size=64, activation='relu'):
        self.device = device
        self.num_actions = num_actions
        self.gamma = gamma                  # discount factor
        self.tau = tau                      # soft update rate for target network
        self.batch_size = batch_size

        # Online Q-network (policy) and target Q-network
        self.policy_net = DQNNetwork(num_states, num_actions, hidden_layers, activation).to(device)
        self.target_net = DQNNetwork(num_states, num_actions, hidden_layers, activation).to(device)

        # Initialize target = policy, and keep target in eval mode
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer for training the policy network
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Replay buffer (simple python list)
        self.replay_buffer = []
        self.buffer_size = buffer_size
   
    def select_action(self, state, epsilon, use_softmax=False, temperature=1.5):
        """
        Action selection:
          - With probability epsilon: choose a random action (exploration)
          - Otherwise:
              * If use_softmax=False: choose argmax Q(s, a)
              * If use_softmax=True : sample from softmax(Q/temperature)
                (Boltzmann exploration; higher temperature -> more random)
        """
        # Epsilon-greedy exploration
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)

        # Convert state to tensor (batch of size 1)
        state_tensor = torch.tensor(np.array(state), dtype=torch.float32, device=self.device).unsqueeze(0)

        # Compute Q-values without gradient tracking
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]

        if use_softmax:
            # Softmax sampling:
            # subtract max(q) to avoid numerical overflow in exp
            exp_q = np.exp((q_values - np.max(q_values)) / temperature)
            probs = exp_q / (np.sum(exp_q) + 1e-8)
            return np.random.choice(self.num_actions, p=probs)
        else:
            # Greedy selection
            return int(np.argmax(q_values))

    def store_transition(self, transition):
        """
        Store one transition in replay buffer.
        Transition format expected: (state, action, reward, next_state)
        """
        self.replay_buffer.append(transition)

        # Keep replay buffer within its maximum size
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)

    def sample_batch(self):
        """Uniformly sample a minibatch from replay buffer."""
        return random.sample(self.replay_buffer, self.batch_size)

    def train_step(self):
        """
        One DQN training update:
          - sample minibatch from replay buffer
          - compute TD target using target network
          - minimize MSE between Q(s,a) and target
          - soft-update target network parameters
        """
        # Not enough samples yet
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = self.sample_batch()
        states, actions, rewards, next_states = zip(*batch)

        # Convert to tensors
        # Using np.array(...) ensures consistent tensor shapes for batches
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)

        # Q(s,a) from policy network for the chosen actions
        q_values = self.policy_net(states).gather(1, actions)

        # TD target: r + gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + self.gamma * max_next_q

        # MSE loss between current Q and TD target
        loss = nn.MSELoss()(q_values, target_q)

        # Gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update (Polyak averaging) for target network:
        # target = tau * policy + (1 - tau) * target
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
