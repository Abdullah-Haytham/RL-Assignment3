import torch
from torch import nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes=(128, 128), n_actions: int = 2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.shared = nn.Sequential(*layers)
        self.policy_head = nn.Linear(in_dim, n_actions)
        self.value_head = nn.Linear(in_dim, 1)

    def forward(self, x):
        x = self.shared(x)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value


class A2CAgent:
    """A simple Advantage Actor-Critic (A2C) agent.

    - Uses a shared actor-critic network.
    - Collects transitions per-episode and updates at episode end.
    - Exposes `select_action(state, eps_threshold=0)` to match DQN API (deterministic when eps=0).
    """

    def __init__(self, n_observations, n_actions, device, lr=1e-3, gamma=0.99,
                 entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.5):
        self.device = device
        self.n_actions = n_actions
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        self.actor_critic = ActorCritic(n_observations, n_actions=n_actions).to(device)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr)

        # on-policy buffer for current episode
        self.episode_buffer = []  # list of (log_prob, value, reward, entropy)
        # temporary storage for current step's log_prob, value, entropy (set by select_action)
        self._current_log_prob = None
        self._current_value = None
        self._current_entropy = None

    def select_action(self, state, eps_threshold=None, deterministic=False):
        # state: tensor [1, obs]
        logits, value = self.actor_critic(state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        if deterministic:
            action = probs.argmax(dim=-1, keepdim=True)
            # No gradient needed for deterministic (eval) mode
            self._current_log_prob = None
            self._current_value = None
            self._current_entropy = None
        else:
            action = dist.sample().view(1, 1)
            # Store with gradients for training
            self._current_log_prob = dist.log_prob(action.view(-1))
            self._current_value = value.squeeze(0)
            self._current_entropy = dist.entropy()

        return action.to(self.device)

    def push_transition(self, state, action, next_state, reward):
        # Use log_prob/value/entropy computed during select_action (with gradients)
        if self._current_log_prob is not None:
            reward_tensor = reward if isinstance(reward, torch.Tensor) else torch.tensor([reward], device=self.device)
            # If reward is already a tensor, extract scalar value
            if isinstance(reward, torch.Tensor):
                reward_val = reward.item() if reward.numel() == 1 else reward[0].item()
                reward_tensor = torch.tensor([reward_val], device=self.device)
            self.episode_buffer.append((
                self._current_log_prob,
                self._current_value,
                reward_tensor,
                self._current_entropy
            ))
            # Clear temporary storage
            self._current_log_prob = None
            self._current_value = None
            self._current_entropy = None

    def optimize(self):
        # Run optimization using collected episode buffer. If buffer empty, skip.
        if len(self.episode_buffer) == 0:
            return None

        # compute returns
        returns = []
        R = 0
        for _, _, reward, _ in reversed(self.episode_buffer):
            R = reward + self.gamma * R
            returns.insert(0, R)

        returns = torch.cat(returns).detach()
        log_probs = torch.cat([lp.unsqueeze(0) for lp, _, _, _ in self.episode_buffer])
        values = torch.cat([v.unsqueeze(0) for _, v, _, _ in self.episode_buffer]).squeeze(1)
        entropies = torch.cat([e.unsqueeze(0) for _, _, _, e in self.episode_buffer])

        advantages = returns - values

        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(values, returns)
        entropy_loss = -entropies.mean()

        loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # clear buffer
        self.episode_buffer = []

        return loss.item()

    def save(self, path: str):
        torch.save({'policy_state_dict': self.actor_critic.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(ckpt['policy_state_dict'])
