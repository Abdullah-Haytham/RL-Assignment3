"""Run a trained model for N episodes and log episode durations to WandB and CSV.

Usage:
    python run_tests.py --env CartPole-v1 --model models/CartPole-v1_dqn.pt --episodes 100
"""
import argparse
import csv
import os
from pathlib import Path
from typing import Optional, Union

import gymnasium as gym
from gymnasium import ActionWrapper, spaces
import numpy as np
import torch
import wandb

from rl.agents import A2CAgent, SACAgent


class DiscretizeAction(ActionWrapper):
    """Wrapper to discretize continuous action spaces for agents."""
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
            n_discrete_actions=15, video_frequency=50):
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
            video_path = os.path.join(video_folder, "tests", algo.upper(), env_name)
        else:
            video_path = os.path.join(video_folder, "tests" ,env_name)
        os.makedirs(video_path, exist_ok=True)
        def should_record(episode_id):
            return episode_id == 0 or ((episode_id + 1) % video_frequency == 0)

        env = gym.wrappers.RecordVideo(env, video_path, episode_trigger=should_record, name_prefix=env_name)
    return env


def run_tests(
    env_name: str,
    model_path: str,
    episodes: int = 100,
    device: Optional[str] = None,
    project: str = "rl-dqn",
    entity: Optional[str] = None,
    record_video: bool = False,
    algo: str = "a2c",
):
    device_name = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    torch_device = torch.device(device_name)

    model_name = Path(model_path).stem

    run_name = f"test_{algo}_{env_name}_{episodes}eps"
    run = wandb.init(
        project="rl-ass4",
        entity=entity,
        name=run_name,
        config={
            'env': env_name,
            'model_path': model_path,
            'episodes': episodes,
            'algo': algo,
        }
    )

    env = make_env(env_name, record_video=record_video, algo=algo)
    obs, _ = env.reset()
    obs_array = np.asarray(obs, dtype=np.float32).reshape(-1)
    n_observations = obs_array.shape[0]
    action_space = env.action_space

    if not isinstance(action_space, spaces.Discrete):
        raise ValueError("Testing expects a discrete action space. Consider discretizing continuous actions first.")

    n_actions = int(action_space.n)

    if algo == 'sac':
        agent: Union[A2CAgent, SACAgent] = SACAgent(
            n_observations,
            n_actions,
            device=torch_device,
            lr=3e-4,
            gamma=0.99,
            tau=0.005,
            alpha=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
        )
    else:
        agent = A2CAgent(n_observations, n_actions, device=torch_device)

    agent.load(model_path)

    durations = []
    os.makedirs('results', exist_ok=True)
    csv_path = f'results/{Path(model_path).stem}_test_results.csv'
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['episode', 'length'])
        for i in range(episodes):
            obs, _ = env.reset()
            obs_array = np.asarray(obs, dtype=np.float32).reshape(-1)
            state = torch.tensor(obs_array, dtype=torch.float32, device=torch_device).unsqueeze(0)
            done = False
            t = 0
            while not done:
                with torch.no_grad():
                    action_tensor = agent.select_action(state, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(int(action_tensor.item()))
                done = terminated or truncated
                if not done:
                    obs_array = np.asarray(obs, dtype=np.float32).reshape(-1)
                    state = torch.tensor(obs_array, dtype=torch.float32, device=torch_device).unsqueeze(0)
                t += 1
            durations.append(t)
            writer.writerow([i, t])
            run.log({'test/episode_length': t, 'test/episode': i})

    run.finish()
    env.close()
    print(f"Saved results to {csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--record-video', action='store_true')
    parser.add_argument('--project', type=str, default='rl-dqn')
    parser.add_argument('--entity', type=str, default=None)
    parser.add_argument('--algo', type=str, choices=['a2c', 'sac'], default='a2c')
    args = parser.parse_args()

    run_tests(
        env_name=args.env,
        model_path=args.model,
        episodes=args.episodes,
        record_video=args.record_video,
        project=args.project,
        entity=args.entity,
        algo=args.algo,
    )
