"""Train agents on Gymnasium environments with WandB logging."""
import argparse
import os
import gymnasium as gym
from gymnasium import ActionWrapper, spaces
import numpy as np
import torch
import wandb

from typing import Optional, Union

from rl.agents import A2CAgent, SACAgent


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
    device: Optional[str] = None,
    project: str = "rl-ass4",
    entity: Optional[str] = None,
    seed: int = 42,
    sac_alpha: float = 0.2,
    sac_tau: float = 0.005,
    sac_replay_size: int = 100_000,
    sac_batch_size: int = 64,
    sac_warmup_steps: int = 1000,
    sac_updates_per_optimize: int = 1,
):
    device_name = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    torch_device = torch.device(device_name)
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    run_name = f"{algo}_{env_name}_{episodes}eps"
    config = {
        'env': env_name,
        'algo': algo,
        'episodes': episodes,
        'lr': lr,
        'gamma': gamma,
        'entropy_coef': entropy_coef,
        'value_coef': value_coef,
        'max_grad_norm': max_grad_norm,
        'seed': seed,
    }
    if algo == 'sac':
        config.update({
            'sac_alpha': sac_alpha,
            'sac_tau': sac_tau,
            'sac_replay_size': sac_replay_size,
            'sac_batch_size': sac_batch_size,
            'sac_warmup_steps': sac_warmup_steps,
            'sac_updates_per_optimize': sac_updates_per_optimize,
        })

    run = wandb.init(project=project, entity=entity, name=run_name, config=config)

    env = make_env(env_name, seed=seed, record_video=record_video, algo=algo)
    obs, _ = env.reset(seed=seed)
    obs_array = np.asarray(obs, dtype=np.float32)
    obs_vector = obs_array.reshape(-1)
    n_observations = obs_vector.shape[0]

    action_space = env.action_space
    agent: Union[A2CAgent, SACAgent]
    if isinstance(action_space, spaces.Discrete):
        n_actions = int(action_space.n)
        if algo == 'sac':
            agent = SACAgent(
                n_observations,
                n_actions,
                device=torch_device,
                lr=lr,
                gamma=gamma,
                tau=sac_tau,
                alpha=sac_alpha,
                value_coef=value_coef,
                entropy_coef=entropy_coef,
                max_grad_norm=max_grad_norm,
                replay_size=sac_replay_size,
                batch_size=sac_batch_size,
                warmup_steps=sac_warmup_steps,
                updates_per_optimize=sac_updates_per_optimize,
            )
        else:
            agent = A2CAgent(
                n_observations,
                n_actions,
                device=torch_device,
                lr=lr,
                gamma=gamma,
                entropy_coef=entropy_coef,
                value_coef=value_coef,
                max_grad_norm=max_grad_norm,
            )
    else:
        raise ValueError(
            "Expecting a discrete action space. Consider enabling discretization or choosing a supported environment."
        )

    print(f"Training {algo.upper()} on {env_name} for {episodes} episodes...")
    action_caption = f"Actions: {int(action_space.n)}"
    print(f"Device: {torch_device.type}, {action_caption}, Observations: {n_observations}")
    
    for i_episode in range(episodes):
        obs, _ = env.reset()
        obs_vec = np.asarray(obs, dtype=np.float32).reshape(-1)
        state = torch.tensor(obs_vec, dtype=torch.float32, device=torch_device).unsqueeze(0)
        total_reward = 0.0
        t = 0
        last_metrics = None
        current_obs = obs_vec
        
        while True:
            action_tensor = agent.select_action(state, deterministic=False)
            action_value = int(action_tensor.item())
            next_obs, reward, terminated, truncated, _ = env.step(action_value)
            reward_value = float(reward)
            done = terminated or truncated
            
            next_obs_vec = np.asarray(next_obs, dtype=np.float32).reshape(-1)
            next_state = torch.tensor(next_obs_vec, dtype=torch.float32, device=torch_device).unsqueeze(0)

            if isinstance(agent, SACAgent):
                agent.push_transition(current_obs, action_value, next_obs_vec, reward_value, done)
                step_metrics = agent.optimize()
                if step_metrics is not None:
                    last_metrics = step_metrics
            else:
                agent.push_transition(state, action_tensor, next_state, reward_value)

            state = next_state
            current_obs = next_obs_vec
            total_reward += reward_value
            t += 1

            if done:
                # Optimize at end of episode
                if isinstance(agent, SACAgent):
                    metrics = last_metrics
                else:
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
                    optional_metric_mapping = {
                        'mean_q': 'train/mean_q',
                        'replay_buffer_size': 'train/replay_buffer_size',
                        'actor_entropy': 'train/actor_entropy',
                    }
                    for optional_key, wandb_key in optional_metric_mapping.items():
                        if optional_key in metrics:
                            log_dict[wandb_key] = metrics[optional_key]
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
    parser.add_argument('--algo', type=str, choices=['a2c', 'sac'], default='a2c')
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
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--sac-alpha', type=float, default=0.2)
    parser.add_argument('--sac-tau', type=float, default=0.005)
    parser.add_argument('--sac-replay-size', type=int, default=100_000)
    parser.add_argument('--sac-batch-size', type=int, default=64)
    parser.add_argument('--sac-warmup-steps', type=int, default=1000)
    parser.add_argument('--sac-updates-per-optimize', type=int, default=1)
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
        device=args.device,
        sac_alpha=args.sac_alpha,
        sac_tau=args.sac_tau,
        sac_replay_size=args.sac_replay_size,
        sac_batch_size=args.sac_batch_size,
        sac_warmup_steps=args.sac_warmup_steps,
        sac_updates_per_optimize=args.sac_updates_per_optimize,
    )