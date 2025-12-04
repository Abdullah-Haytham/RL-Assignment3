import random
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class ReplayBuffer:
    """Simple FIFO replay buffer for off-policy agents."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: Deque[Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]] = deque(maxlen=capacity)

    def push(
        self,
        state: Union[Sequence[float], np.ndarray, float],
        action: Union[Sequence[float], np.ndarray, float],
        reward: float,
        next_state: Union[Sequence[float], np.ndarray, float],
        done: bool,
    ) -> None:
        state_arr = np.asarray(state, dtype=np.float32)
        action_arr = np.asarray(action)
        next_state_arr = np.asarray(next_state, dtype=np.float32)
        self.buffer.append((state_arr, action_arr, float(reward), next_state_arr, float(done)))

    def sample(self, batch_size: int) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, List[np.ndarray], np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return list(states), list(actions), np.asarray(rewards, dtype=np.float32), list(next_states), np.asarray(dones, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.buffer)


class A2CAgent:
    """Advantage Actor-Critic agent for discrete action spaces."""

    def __init__(
        self,
        n_observations: int,
        n_actions: int,
        device: torch.device,
        lr: float = 3e-4,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
    ) -> None:
        self.device = device
        self.n_actions = n_actions
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        hidden_size = 128
        self.actor = nn.Sequential(
            nn.Linear(n_observations, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        ).to(self.device)

        self.critic = nn.Sequential(
            nn.Linear(n_observations, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr,
        )

        self.rollout: List[Dict[str, torch.Tensor]] = []
        self.uses_replay_buffer = False

    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        else:
            state = state.to(self.device)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        logits = self.actor(state)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        return action

    def push_transition(self, state, action, next_state, reward, done: bool = False) -> None:  # noqa: D401 signature
        _ = next_state
        _ = done
        state_tensor = state.detach().to(self.device).squeeze(0)
        if isinstance(action, torch.Tensor):
            action_tensor = action.detach().to(self.device)
        else:
            action_tensor = torch.tensor(action, dtype=torch.long, device=self.device)
        reward_tensor = torch.tensor(float(reward), dtype=torch.float32, device=self.device)
        self.rollout.append({
            "state": state_tensor,
            "action": action_tensor.squeeze(-1) if action_tensor.dim() > 0 else action_tensor,
            "reward": reward_tensor,
        })

    def optimize(self, last_state: Optional[torch.Tensor] = None) -> Optional[Dict[str, Any]]:
        if not self.rollout:
            return None

        states = torch.stack([entry["state"] for entry in self.rollout]).to(self.device)
        actions = torch.stack([entry["action"] for entry in self.rollout]).to(self.device)
        rewards = torch.stack([entry["reward"] for entry in self.rollout]).to(self.device)

        with torch.no_grad():
            if last_state is not None:
                bootstrap_value = self.critic(last_state.to(self.device)).squeeze(-1)
            else:
                bootstrap_value = torch.zeros(1, device=self.device)

            returns: List[torch.Tensor] = []
            running_return = bootstrap_value.squeeze(-1)
            for reward in reversed(rewards):
                running_return = reward + self.gamma * running_return
                returns.append(running_return)
            returns_tensor = torch.stack(list(reversed(returns)))

        values = self.critic(states).squeeze(-1)
        advantages = returns_tensor - values

        dist = torch.distributions.Categorical(logits=self.actor(states))
        log_probs = dist.log_prob(actions.long())
        entropy = dist.entropy().mean()

        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = 0.5 * advantages.pow(2).mean()
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            self.max_grad_norm,
        )
        self.optimizer.step()

        metrics = {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "advantages_mean": float(advantages.mean().item()),
            "advantages_std": float(advantages.std(unbiased=False).item()),
            "grad_norm": float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
        }

        self.rollout.clear()
        return metrics

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


class SACAgent:
    """Soft Actor-Critic (SAC) Agent implementation for discrete action spaces."""

    def __init__(
        self,
        n_observations: int,
        n_actions: int,
        device: torch.device,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        replay_size: int = 100_000,
        batch_size: int = 64,
        warmup_steps: int = 1000,
        updates_per_optimize: int = 1,
    ) -> None:
        self.device = device
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.updates_per_optimize = updates_per_optimize
        self.total_steps = 0
        self.total_updates = 0
        self.replay_buffer = ReplayBuffer(replay_size)
        self.uses_replay_buffer = True

        self.actor = nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        ).to(self.device)

        critic_input_dim = n_observations + n_actions
        self.critic1 = nn.Sequential(
            nn.Linear(critic_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        ).to(self.device)
        self.critic2 = nn.Sequential(
            nn.Linear(critic_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        ).to(self.device)

        self.target_critic1 = nn.Sequential(
            nn.Linear(critic_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        ).to(self.device)
        self.target_critic2 = nn.Sequential(
            nn.Linear(critic_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        ).to(self.device)

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr)

    def _prepare_state(self, array_like) -> np.ndarray:
        return np.asarray(array_like, dtype=np.float32)

    def _prepare_action(self, action: Union[torch.Tensor, int, np.ndarray]) -> int:
        if isinstance(action, torch.Tensor):
            return int(action.item())
        if isinstance(action, np.ndarray):
            return int(action.item())
        return int(action)

    def push_transition(self, state, action, next_state, reward, done: bool = False) -> None:
        state_np = self._prepare_state(state)
        next_state_np = self._prepare_state(next_state)
        action_idx = self._prepare_action(action)
        reward_val = float(reward)
        self.replay_buffer.push(state_np, action_idx, reward_val, next_state_np, bool(done))
        self.total_steps += 1

    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        else:
            state = state.to(self.device)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        logits = self.actor(state)
        probs = F.softmax(logits, dim=-1)

        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

        return action

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def _evaluate_all_actions(self, critic: nn.Module, states: torch.Tensor) -> torch.Tensor:
        batch_size = states.shape[0]
        action_eye = torch.eye(self.n_actions, device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
        state_expanded = states.unsqueeze(1).expand(-1, self.n_actions, -1)
        critic_input = torch.cat([state_expanded, action_eye], dim=-1)
        critic_input = critic_input.view(batch_size * self.n_actions, -1)
        q_values = critic(critic_input)
        return q_values.view(batch_size, self.n_actions)

    def optimize(self, last_state: Optional[torch.Tensor] = None) -> Optional[Dict[str, Any]]:
        _ = last_state
        if len(self.replay_buffer) < self.batch_size:
            return None
        if self.total_steps < self.warmup_steps:
            return None

        metrics_accumulator: List[Dict[str, float]] = []
        updates = self.updates_per_optimize
        eps = 1e-8

        for _ in range(updates):
            if len(self.replay_buffer) < self.batch_size:
                break

            states_np, actions, rewards, next_states_np, dones = self.replay_buffer.sample(self.batch_size)
            actions_array = np.asarray(actions, dtype=np.int64)
            states = torch.tensor(np.stack(states_np), dtype=torch.float32, device=self.device)
            next_states = torch.tensor(np.stack(next_states_np), dtype=torch.float32, device=self.device)
            actions_tensor = torch.tensor(actions_array, dtype=torch.long, device=self.device).unsqueeze(-1)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
            dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)

            action_one_hot = F.one_hot(actions_tensor.squeeze(-1), num_classes=self.n_actions).float()
            critic_input = torch.cat([states, action_one_hot], dim=-1)
            current_q1 = self.critic1(critic_input)
            current_q2 = self.critic2(critic_input)

            with torch.no_grad():
                next_logits = self.actor(next_states)
                next_probs = F.softmax(next_logits, dim=-1)
                next_log_probs = torch.log(next_probs + eps)
                target_q1 = self._evaluate_all_actions(self.target_critic1, next_states)
                target_q2 = self._evaluate_all_actions(self.target_critic2, next_states)
                target_min_q = torch.min(target_q1, target_q2)
                next_value = (next_probs * (target_min_q - self.alpha * next_log_probs)).sum(dim=-1, keepdim=True)
                target_q = rewards_tensor + self.gamma * (1.0 - dones_tensor) * next_value

            critic1_loss = F.mse_loss(current_q1, target_q)
            critic2_loss = F.mse_loss(current_q2, target_q)
            critic_loss = critic1_loss + critic2_loss

            self.critic1_optimizer.zero_grad()
            self.critic2_optimizer.zero_grad()
            critic_loss.backward()
            critic1_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), self.max_grad_norm)
            critic2_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.max_grad_norm)
            self.critic1_optimizer.step()
            self.critic2_optimizer.step()

            probs = F.softmax(self.actor(states), dim=-1)
            log_probs = torch.log(probs + eps)
            q1_values = self._evaluate_all_actions(self.critic1, states).detach()
            q2_values = self._evaluate_all_actions(self.critic2, states).detach()
            min_q = torch.min(q1_values, q2_values)
            actor_loss = (probs * (self.alpha * log_probs - min_q)).sum(dim=-1).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            self._soft_update(self.target_critic1, self.critic1)
            self._soft_update(self.target_critic2, self.critic2)

            entropy = -(probs * log_probs).sum(dim=-1).mean().item()
            avg_grad_norm = float(
                np.mean(
                    [
                        actor_grad_norm.item() if isinstance(actor_grad_norm, torch.Tensor) else actor_grad_norm,
                        critic1_grad_norm.item() if isinstance(critic1_grad_norm, torch.Tensor) else critic1_grad_norm,
                        critic2_grad_norm.item() if isinstance(critic2_grad_norm, torch.Tensor) else critic2_grad_norm,
                    ]
                )
            )

            metrics_accumulator.append(
                {
                    "loss": critic_loss.item() + actor_loss.item(),
                    "policy_loss": actor_loss.item(),
                    "value_loss": 0.5 * (critic1_loss.item() + critic2_loss.item()),
                    "entropy": entropy,
                    "actor_entropy": entropy,
                    "advantages_mean": 0.0,
                    "advantages_std": 0.0,
                    "grad_norm": avg_grad_norm,
                    "mean_q": min_q.mean().item(),
                    "replay_buffer_size": float(len(self.replay_buffer)),
                }
            )

            self.total_updates += 1

        if not metrics_accumulator:
            return None

        aggregated: Dict[str, float] = {}
        for key in metrics_accumulator[0].keys():
            aggregated[key] = float(np.mean([m[key] for m in metrics_accumulator]))
        return aggregated

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic1_state_dict": self.critic1.state_dict(),
                "critic2_state_dict": self.critic2.state_dict(),
                "target_critic1_state_dict": self.target_critic1.state_dict(),
                "target_critic2_state_dict": self.target_critic2.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic1_optimizer_state_dict": self.critic1_optimizer.state_dict(),
                "critic2_optimizer_state_dict": self.critic2_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.critic1.load_state_dict(ckpt["critic1_state_dict"])
        self.critic2.load_state_dict(ckpt["critic2_state_dict"])
        self.target_critic1.load_state_dict(ckpt["target_critic1_state_dict"])
        self.target_critic2.load_state_dict(ckpt["target_critic2_state_dict"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer_state_dict"])
        self.critic1_optimizer.load_state_dict(ckpt["critic1_optimizer_state_dict"])
        self.critic2_optimizer.load_state_dict(ckpt["critic2_optimizer_state_dict"])

        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(param.data)

