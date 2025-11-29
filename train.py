"""Train agents on Gymnasium environments with WandB logging.

This script supports `dqn`, `ddqn`, and `a2c` (uses the A2C agent in `rl.agents`).
"""
import argparse
import os
import time
from pathlib import Path

import gymnasium as gym
from gymnasium import ActionWrapper, spaces
import numpy as np
import torch
import wandb
import math

from rl.agents import A2CAgent


class DiscretizeAction(ActionWrapper):
    """Wrapper to discretize continuous action spaces for DQN/A2C."""
    def __init__(self, env, n_actions=11):
        super().__init__(env)
        self.n_actions = n_actions
        self.action_space = spaces.Discrete(n_actions)
        if hasattr(env.action_space, 'shape') and len(env.action_space.shape) > 0:
            self.action_dim = env.action_space.shape[0]
            self.continuous_actions = []
            for i in range(self.action_dim):
                low = env.action_space.low[i]
                high = env.action_space.high[i]
                self.continuous_actions.append(np.linspace(low, high, n_actions))
        else:
            self.action_dim = 1
            low = env.action_space.low[0] if hasattr(env.action_space.low, '__getitem__') else env.action_space.low
            high = env.action_space.high[0] if hasattr(env.action_space.high, '__getitem__') else env.action_space.high
            self.continuous_actions = [np.linspace(low, high, n_actions)]

    def action(self, action):
        if self.action_dim == 1:
            return np.array([self.continuous_actions[0][action]])
        else:
            return np.array([self.continuous_actions[i][action] for i in range(self.action_dim)])


def make_env(env_name, seed=None, record_video=False, video_folder="videos", algo=None,
             n_discrete_actions=11, video_frequency=50):
    render_mode = 'rgb_array' if record_video else None
    env = gym.make(env_name, render_mode=render_mode)
    if isinstance(env.action_space, spaces.Box):
        print(f"Detected continuous action space for {env_name}. Discretizing into {n_discrete_actions} actions.")
        env = DiscretizeAction(env, n_actions=n_discrete_actions)
    if seed is not None:
        try:
            env.reset(seed=seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
        except Exception:
            pass
    if record_video:
        if algo:
            video_path = os.path.join(video_folder, algo.upper(), env_name)
        else:
            video_path = os.path.join(video_folder, env_name)
        os.makedirs(video_path, exist_ok=True)
        def should_record(episode_id):
            return episode_id == 0 or ((episode_id + 1) % video_frequency == 0)
        env = gym.wrappers.RecordVideo(env, video_path, episode_trigger=should_record, name_prefix=env_name)
    return env


def train(
    env_name: str,
    algo: str,
    episodes: int = 200,
    lr: float = 1e-3,
    gamma: float = 0.98,
    entropy_coef: float = 0.02,
    value_coef: float = 0.7,
    max_grad_norm: float = 1.0,
    record_video: bool = False,
    device: str = None,
    project: str = "rl-ass4",
    entity: str = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    run_name = f"{algo}_{env_name}_{episodes}eps"
    run = wandb.init(project=project, entity=entity, name=run_name, config={
        'env': env_name,
        'algo': algo,
        'episodes': episodes,
        'lr': lr,
        'gamma': gamma,
        'entropy_coef': entropy_coef,
        'value_coef': value_coef,
        'max_grad_norm': max_grad_norm,
    })
    cfg = run.config

    env = make_env(env_name, record_video=record_video, algo=algo)
    n_actions = env.action_space.n
    obs, _ = env.reset()
    n_observations = len(obs)

    agent = A2CAgent(n_observations, n_actions, device=device, lr=lr, gamma=gamma, entropy_coef=entropy_coef, value_coef=value_coef, max_grad_norm=max_grad_norm)

    episode_durations = []

    print(f"Training {algo.upper()} on {env_name} for {episodes} episodes...")
    
    for i_episode in range(episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0.0
        t = 0
        
        print(f"\rEpisode {i_episode + 1}/{episodes} - Running...", end='', flush=True)
        
        while True:
            action = agent.select_action(state, deterministic=False)

            obs, reward, terminated, truncated, _ = env.step(int(action.item()))
            done = terminated or truncated
            reward_t = torch.tensor([reward], device=device)
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            agent.push_transition(state, action, next_state, reward_t)
            state = next_state
            total_reward += float(reward)
            t += 1

            if done:
                loss = agent.optimize()
                if loss is not None:
                    run.log({'train/loss': loss, 'train/episode': i_episode})

                episode_durations.append(t)
                run.log({'train/episode_reward': total_reward, 'train/episode_length': t, 'train/episode': i_episode})
                print(f"\rEpisode {i_episode + 1}/{episodes} - Reward: {total_reward:.2f}, Length: {t}", end='', flush=True)
                break

        if (i_episode + 1) % 50 == 0:
            os.makedirs('models', exist_ok=True)
            model_path = f"models/{env_name}_{algo}.pt"
            agent.save(model_path)
            print()

    print(f"\n\nTraining completed! Final model saved.")
    os.makedirs('models', exist_ok=True)
    model_path = f"models/{env_name}_{algo}.pt"
    agent.save(model_path)
    run.finish()
    env.close()


if __name__ == '__main__':
    import math

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--algo', type=str, choices=['a2c'], default='a2c')
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--learning-rate', '--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.98)
    parser.add_argument('--entropy-coef', type=float, default=0.02)
    parser.add_argument('--value-coef', type=float, default=0.7)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--record-video', action='store_true')
    parser.add_argument('--project', type=str, default='rl-dqn')
    parser.add_argument('--entity', type=str, default=None)
    args = parser.parse_args()

    train(env_name=args.env, algo=args.algo, episodes=args.episodes, lr=args.learning_rate, gamma=args.gamma, entropy_coef=args.entropy_coef, value_coef=args.value_coef, max_grad_norm=args.max_grad_norm, record_video=args.record_video, project=args.project, entity=args.entity)
