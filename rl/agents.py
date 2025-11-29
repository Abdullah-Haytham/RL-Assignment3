import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
import numpy as np


class A2CAgent:
    """
    A2C Agent implementation based on working reference implementation.
    Uses separate Actor and Critic networks.
    """
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
    ):
        self.device = device
        self.n_actions = n_actions
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        # Separate Actor and Critic networks (like working implementation)
        self.actor = nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        ).to(self.device)
        
        self.critic = nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(self.device)
        
        # Separate optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Episode buffers
        self.log_probs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.entropies: List[torch.Tensor] = []

    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Select action and store log_prob, value, entropy for training."""
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        else:
            state = state.to(self.device)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Get action probabilities from actor
        logits = self.actor(state)
        probs = F.softmax(logits, dim=-1)
        
        # Get value from critic
        value = self.critic(state)
        
        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            # Sample action
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            
            # Store for training (keep gradients)
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            
            self.log_probs.append(log_prob)
            self.values.append(value.squeeze())
            self.entropies.append(entropy)

        return action

    def push_transition(self, state, action, next_state, reward):
        """Store reward."""
        if isinstance(reward, torch.Tensor):
            self.rewards.append(float(reward.item()))
        else:
            self.rewards.append(float(reward))

    def optimize(self, last_state: Optional[torch.Tensor] = None) -> Optional[Dict[str, Any]]:
        """Perform A2C update."""
        if len(self.log_probs) == 0:
            return None

        # Compute returns
        R = 0.0
        if last_state is not None:
            with torch.no_grad():
                if not isinstance(last_state, torch.Tensor):
                    last_state = torch.tensor(last_state, dtype=torch.float32, device=self.device)
                else:
                    last_state = last_state.to(self.device)
                
                if last_state.dim() == 1:
                    last_state = last_state.unsqueeze(0)
                
                R = self.critic(last_state).squeeze().item()
        
        returns = []
        for reward in reversed(self.rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # Stack stored tensors
        log_probs = torch.stack(self.log_probs)
        values = torch.stack(self.values)
        entropies = torch.stack(self.entropies)
        
        # Compute advantages
        advantages = returns - values.detach()
        
        # Compute losses
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = F.mse_loss(values, returns)
        entropy_loss = -entropies.mean()
        
        # Total loss
        loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
        
        # Optimize
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), 
            self.max_grad_norm
        )
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), 
            self.max_grad_norm
        )
        
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        
        # Metrics
        metrics = {
            "loss": loss.item(),
            "policy_loss": actor_loss.item(),
            "value_loss": critic_loss.item(),
            "entropy": entropies.mean().item(),
            "advantages_mean": advantages.mean().item(),
            "advantages_std": advantages.std().item() if len(advantages) > 1 else 0.0,
            "grad_norm": (actor_grad_norm.item() if isinstance(actor_grad_norm, torch.Tensor) else actor_grad_norm),
            "mean_value": values.mean().item(),
            "mean_return": returns.mean().item(),
        }
        
        # Clear buffers
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropies = []
        
        return metrics

    def save(self, path: str):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        # Support old format (policy_state_dict) and new format (actor/critic)
        if 'policy_state_dict' in ckpt:
            # Old format: load weights into both actor and critic if possible
            self.actor.load_state_dict(ckpt['policy_state_dict'], strict=False)
            self.critic.load_state_dict(ckpt['policy_state_dict'], strict=False)
        else:
            self.actor.load_state_dict(ckpt['actor_state_dict'])
            self.critic.load_state_dict(ckpt['critic_state_dict'])
            if 'actor_optimizer_state_dict' in ckpt:
                self.actor_optimizer.load_state_dict(ckpt['actor_optimizer_state_dict'])
            if 'critic_optimizer_state_dict' in ckpt:
                self.critic_optimizer.load_state_dict(ckpt['critic_optimizer_state_dict'])