"""Inspect policy probabilities without changing the checkpoint."""
import argparse
import json

import numpy as np
import torch
from gymnasium.wrappers import FrameStackObservation
from stable_baselines3 import PPO

from rocket_env import RocketLandingEnv


def analyze(checkpoint, episodes, challenge, device):
    env = FrameStackObservation(RocketLandingEnv(challenge=challenge), 8)
    model = PPO.load(checkpoint, device=device)
    probability_sum = np.zeros(4, dtype=np.float64)
    argmax_counts = np.zeros(4, dtype=np.int64)
    sampled_states = 0
    entropies = []

    with torch.no_grad():
        for episode in range(episodes):
            obs, _ = env.reset(seed=910_000 + episode)
            while True:
                obs_tensor, _ = model.policy.obs_to_tensor(obs)
                distribution = model.policy.get_distribution(obs_tensor).distribution
                probabilities = distribution.probs[0].detach().cpu().numpy()
                probability_sum += probabilities
                argmax_counts[int(np.argmax(probabilities))] += 1
                entropies.append(float(distribution.entropy()[0].detach().cpu()))
                sampled_states += 1
                action = int(np.argmax(probabilities))
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
    env.close()
    return {
        "checkpoint": checkpoint,
        "episodes": episodes,
        "challenge": challenge,
        "sampled_states": sampled_states,
        "mean_action_probabilities": (probability_sum / sampled_states).tolist(),
        "argmax_action_counts": argmax_counts.tolist(),
        "mean_entropy": float(np.mean(entropies)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--challenge", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    for checkpoint in args.checkpoints:
        print(json.dumps(
            analyze(checkpoint, args.episodes, args.challenge, args.device),
            indent=2,
        ))


if __name__ == "__main__":
    main()
