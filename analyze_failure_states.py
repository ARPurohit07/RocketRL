"""Inspect which randomized initial states lead to each terminal outcome."""
import argparse
from collections import Counter, defaultdict

import numpy as np
from gymnasium.wrappers import FrameStackObservation
from stable_baselines3 import PPO

from rocket_env import RocketLandingEnv
from rocket_eval import _failure_category


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["outcome"]].append(row)
    for outcome, values in sorted(grouped.items()):
        def mean_abs(key):
            return float(np.mean([abs(value[key]) for value in values]))
        def mean_signed(key):
            return float(np.mean([value[key] for value in values]))
        print(
            f"{outcome:28s} n={len(values):3d} "
            f"start_x={mean_signed('start_x'):+.2f} "
            f"|start_x|={mean_abs('start_x'):.2f} "
            f"height={mean_abs('height'):.2f} "
            f"|start_vx|={mean_abs('start_vx'):.2f} "
            f"descent={mean_abs('start_vy'):.2f} "
            f"|angle|={mean_abs('start_angle'):.3f} "
            f"end_x={mean_signed('end_x'):+.2f} "
            f"|end_x|={mean_abs('end_x'):.2f} "
            f"|end_vx|={mean_abs('end_vx'):.2f}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="rocket_pythia_phase2_best.zip")
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--seed-start", type=int, default=920_000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model = PPO.load(args.checkpoint, device=args.device)
    env = FrameStackObservation(RocketLandingEnv(challenge=1.0), 8)
    rows = []
    action_counts = Counter()
    try:
        for seed in range(args.seed_start, args.seed_start + args.episodes):
            obs, reset_info = env.reset(seed=seed)
            total_reward = 0.0
            while True:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
                action_counts[action] += 1
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                if terminated or truncated:
                    break
            start = reset_info["start"]
            outcome = "landed" if info.get("is_success") else _failure_category(info)
            rows.append({
                "outcome": outcome,
                "reward": total_reward,
                "start_x": float(start["x"]),
                "height": float(start["y"]),
                "start_vx": float(start["vx"]),
                "start_vy": float(start["vy"]),
                "start_angle": float(start["angle"]),
                "end_x": float(info["x"]),
                "end_vx": float(info["vx"]),
            })
    finally:
        env.close()

    summarize(rows)
    print(f"actions={dict(sorted(action_counts.items()))}")


if __name__ == "__main__":
    main()
