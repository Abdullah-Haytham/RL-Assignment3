from os import path
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
import numpy as np
import torch.nn as nn
from torch.distributions import Categorical
from typing import Optional, Dict, Any, List


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

class PPOAgent:
    def __init__(self, n_observations, n_actions, device, lr=3e-4, gamma=0.99, 
                 eps_clip=0.2, K_epochs=10, entropy_coef=0.01, value_coef=0.5, 
                 max_grad_norm=0.5, gae_lambda=0.95):
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs  # Number of PPO update epochs per batch
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda

        # Actor Network (Policy)
        self.actor = nn.Sequential(
            nn.Linear(n_observations, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions)
        ).to(device)

        # Critic Network (Value)
        self.critic = nn.Sequential(
            nn.Linear(n_observations, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        ).to(device)

        # Joint optimizer for simplicity, or separate optimizers
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr},
            {'params': self.critic.parameters(), 'lr': lr}
        ])

        # Rollout Buffer
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.next_values = [] # For GAE
    def select_action(self, state, deterministic=False):

        # Convert state to tensor
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Forward pass
        logits = self.actor(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        # Sample action
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = dist.sample()

        # Store detached rollout data
        self.states.append(state.detach())
        self.actions.append(action.detach())
        self.log_probs.append(dist.log_prob(action).detach())


        return action.item()   # Return as int
    
    def push_transition(self, state, action, next_state, reward):
       # Only reward is needed because state/action/log_prob were stored in select_action
       self.rewards.append(float(reward))

    
    def optimize(self, last_state=None):
        if len(self.states) == 0:
            return None
        # -----------------------
        # 1. Convert buffers to tensors
        # -----------------------
        states = torch.cat(self.states).to(self.device)
        actions = torch.stack(self.actions).squeeze().to(self.device)
        old_log_probs = torch.stack(self.log_probs).squeeze().to(self.device)
        with torch.no_grad():
            values = self.critic(states).squeeze()
        # -----------------------
        # 2. Compute Advantages with GAE
        # -----------------------
        advantages = []
        returns = []
        gae = 0
        # Bootstrap if last_state provided
        if last_state is not None:
            if not isinstance(last_state, torch.Tensor):
                last_state = torch.tensor(last_state, dtype=torch.float32, device=self.device)
            if last_state.dim() == 1:
                last_state = last_state.unsqueeze(0)
            next_value = self.critic(last_state).item()
        else:
            next_value = 0
        # GAE loop
        for t in reversed(range(len(self.rewards))):
            next_val = next_value if t == len(self.rewards) - 1 else values[t+1].item()
            delta = self.rewards[t] + self.gamma * next_val - values[t].item()
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t].item())
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        # Normalize
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # -----------------------
        # 3. PPO Update
        # -----------------------
        for _ in range(self.K_epochs):
            logits = self.actor(states)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            state_values = self.critic(states).squeeze()
            ratios = torch.exp(new_log_probs - old_log_probs)
            # Clipped surrogate objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(state_values, returns)
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
        # -----------------------
        # 4. Clear Buffers
        # -----------------------
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        # -----------------------
        # 5. Return Metrics
        # -----------------------
        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'advantages_mean': advantages.mean().item(),
            'advantages_std': advantages.std().item(),
            'grad_norm': self.max_grad_norm
        }
        
    def save(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
