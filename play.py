import argparse
import time
import torch
import gymnasium as gym

from rl.agents import A2CAgent, PPOAgent


def load_agent(env, algo, device, model_path):
    """Auto-detect obs/action space and load correct agent."""

    obs, _ = env.reset()

    # Observation size
    n_observations = len(obs)

    # Action size (discrete or continuous-discretized)
    if isinstance(env.action_space, gym.spaces.Discrete):
        n_actions = env.action_space.n
    else:
        # Your training wrapper uses 11 discrete bins for continuous envs
        n_actions = 11

    # Load correct agent class
    if algo == "a2c":
        agent = A2CAgent(n_observations, n_actions, device=device)
    elif algo == "ppo":
        agent = PPOAgent(n_observations, n_actions, device=device)
    else:
        raise ValueError("Unknown algorithm")

    # Load weights
    agent.load(model_path)
    print(f"[INFO] Loaded model from {model_path}")

    return agent


def play(env_name, algo, model_path, episodes=3):
    """Runs trained agent visually on ANY of the 4 environments."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use human render so the game window appears
    env = gym.make(env_name, render_mode="human")

    agent = load_agent(env, algo, device, model_path)

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

            # Deterministic action (no randomness)
            action = agent.select_action(state, deterministic=True)

            obs, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated

            total_reward += reward

            time.sleep(0.02)  # slows down animation for visibility

        print(f"[PLAY] Episode {ep+1}/{episodes} — Reward = {total_reward}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, required=True,
                        help="Environment name: CartPole-v1, Acrobot-v1, MountainCar-v0, Pendulum-v1")

    parser.add_argument("--algo", type=str, required=True, choices=["a2c", "ppo"],
                        help="Algorithm used to train the model")

    parser.add_argument("--model", type=str, required=True,
                        help="Path to saved model .pt file")

    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to visually play")

    args = parser.parse_args()

    play(args.env, args.algo, args.model, episodes=args.episodes)
