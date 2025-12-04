"""Train agents on Gymnasium environments with WandB logging."""
import argparse
import os
import gymnasium as gym
from gymnasium import ActionWrapper, spaces
import numpy as np
import torch
import wandb

from rl.agents import A2CAgent, PPOAgent


class DiscretizeAction(ActionWrapper):
    """Wrapper to discretize continuous action spaces."""
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
        print(f"Discretizing action space into {n_discrete_actions} actions.")
        env = DiscretizeAction(env, n_actions=n_discrete_actions)
    if seed is not None:
        try:
            env.reset(seed=seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
        except Exception:
            pass
    if record_video:
        video_path = os.path.join(video_folder, algo.upper() if algo else "", env_name)
        os.makedirs(video_path, exist_ok=True)
        def should_record(episode_id):
            return episode_id == 0 or ((episode_id + 1) % video_frequency == 0)
        env = gym.wrappers.RecordVideo(env, video_path, episode_trigger=should_record, name_prefix=env_name)
    return env


def train(
    env_name: str,
    algo: str,
    episodes: int = 500,
    lr: float = 3e-4,
    gamma: float = 0.99,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    record_video: bool = False,
    device: str = None,
    project: str = "rl-ass4",
    entity: str = None,
    seed: int = 42,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

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
        'seed': seed,
    })

    env = make_env(env_name, seed=seed, record_video=record_video, algo=algo)
    n_actions = env.action_space.n
    obs, _ = env.reset(seed=seed)
    n_observations = len(obs)

    if algo.lower() == "a2c":
        agent = A2CAgent(
            n_observations, 
            n_actions, 
            device=device, 
            lr=lr, 
            gamma=gamma, 
            entropy_coef=entropy_coef, 
            value_coef=value_coef, 
            max_grad_norm=max_grad_norm
        )
    elif algo.lower() == "ppo":
        agent = PPOAgent(
            n_observations, 
            n_actions, 
            device=device, 
            lr=lr, 
            gamma=gamma, 
            entropy_coef=entropy_coef, 
            value_coef=value_coef, 
            max_grad_norm=max_grad_norm
        )


    print(f"Training {algo.upper()} on {env_name} for {episodes} episodes...")
    print(f"Device: {device}, Actions: {n_actions}, Observations: {n_observations}")
    
    for i_episode in range(episodes):
        obs, _ = env.reset()
        state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0.0
        t = 0
        
        while True:
            # Select action (no gradients needed during rollout)
            action = agent.select_action(state, deterministic=False)
            
            # Take step in environment
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            next_state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            agent.push_transition(state, action, next_state, reward)
            
            state = next_state
            total_reward += reward
            t += 1

            if done:
                # Optimize at end of episode
                if truncated:
                    # Bootstrap from last state if truncated
                    metrics = agent.optimize(last_state=state)
                else:
                    # No bootstrap if terminated naturally
                    metrics = agent.optimize(last_state=None)
                
                # Log metrics
                if metrics is not None:
                    log_dict = {
                        'train/loss': metrics['loss'],
                        'train/policy_loss': metrics['policy_loss'],
                        'train/value_loss': metrics['value_loss'],
                        'train/entropy': metrics['entropy'],
                        'train/advantages_mean': metrics['advantages_mean'],
                        'train/advantages_std': metrics['advantages_std'],
                        'train/grad_norm': metrics['grad_norm'],
                        'train/episode_reward': total_reward,
                        'train/episode_length': t,
                        'train/episode': i_episode
                    }
                    run.log(log_dict)
                
                # Print progress
                if (i_episode + 1) % 10 == 0:
                    loss_str = f"{metrics['loss']:.3f}" if metrics else "N/A"
                    entropy_str = f"{metrics['entropy']:.3f}" if metrics else "N/A"
                    print(f"Episode {i_episode + 1}/{episodes} - Reward: {total_reward:.1f}, Length: {t}, "
                          f"Loss: {loss_str}, Entropy: {entropy_str}")
                
                break

        # Save model periodically
        if (i_episode + 1) % 100 == 0:
            os.makedirs('models', exist_ok=True)
            model_path = f"models/{env_name}_{algo}_ep{i_episode+1}.pt"
            agent.save(model_path)

    # Final save
    print(f"\nTraining completed!")
    os.makedirs('models', exist_ok=True)
    model_path = f"models/{env_name}_{algo}_final.pt"
    agent.save(model_path)
    print(f"Model saved to {model_path}")
    
    run.finish()
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--algo', type=str, choices=['a2c', 'ppo'], default='a2c')
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--learning-rate', '--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--entropy-coef', type=float, default=0.01)
    parser.add_argument('--value-coef', type=float, default=0.5)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--record-video', action='store_true')
    parser.add_argument('--project', type=str, default='rl-ass4')
    parser.add_argument('--entity', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    train(
        env_name=args.env,
        algo=args.algo,
        episodes=args.episodes,
        lr=args.learning_rate,
        gamma=args.gamma,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        record_video=args.record_video,
        project=args.project,
        entity=args.entity,
        seed=args.seed,
    )