"""One reproducible evaluator for the 3D task."""

import argparse
from collections import Counter

import numpy as np
from gymnasium.wrappers import FrameStackObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from rocket3d_env import Rocket3DEnv


class Seeded3DEnv(Rocket3DEnv):
    def __init__(self, seeds, worker, workers, challenge):
        super().__init__(challenge=challenge)
        self.seeds = seeds
        self.index = worker
        self.workers = workers

    def reset(self, *, seed=None, options=None):
        assigned = self.seeds[self.index % len(self.seeds)]
        self.index += self.workers
        return super().reset(seed=assigned, options=options)


def evaluate(model, seeds, challenge=1.0, workers=8, deterministic=True):
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("At least one seed is required")
    count = min(workers, len(seeds))
    env = DummyVecEnv([
        (lambda i=i: FrameStackObservation(
            Seeded3DEnv(seeds, i, count, challenge), 8
        )) for i in range(count)
    ])
    quotas = [len(range(i, len(seeds), count)) for i in range(count)]
    completed = [0] * count
    reward_totals = np.zeros(count)
    records = []
    try:
        obs = env.reset()
        while any(done < quota for done, quota in zip(completed, quotas)):
            actions, _ = model.predict(obs, deterministic=deterministic)
            obs, rewards, dones, infos = env.step(actions)
            reward_totals += rewards
            for i, done in enumerate(dones):
                if not done:
                    continue
                if completed[i] < quotas[i]:
                    info = infos[i]
                    records.append({
                        "seed": int(seeds[i + completed[i] * count]),
                        "success": bool(info["is_success"]),
                        "reward": float(reward_totals[i]),
                        "status": info["status"],
                        "reason": info["reason"],
                        "height": float(info["start"]["position"][2]),
                    })
                    completed[i] += 1
                reward_totals[i] = 0
    finally:
        env.close()
    failures = Counter(row["reason"] for row in records if not row["success"])
    height_bands = {}
    for lo, hi in ((6, 10), (10, 16), (16, 24)):
        band = [row for row in records if lo <= row["height"] < hi]
        if band:
            height_bands[f"{lo}-{hi}m"] = sum(row["success"] for row in band) / len(band)
    return {
        "episodes": len(records),
        "landings": sum(row["success"] for row in records),
        "success_rate": np.mean([row["success"] for row in records]).item(),
        "mean_reward": np.mean([row["reward"] for row in records]).item(),
        "failures": dict(failures),
        "by_height": height_bands,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/improved.zip")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=2_000_000)
    parser.add_argument("--challenge", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()
    model = PPO.load(args.checkpoint, device=args.device)
    stats = evaluate(
        model, range(args.seed_start, args.seed_start + args.episodes),
        challenge=args.challenge, deterministic=not args.stochastic,
    )
    print(stats, flush=True)


if __name__ == "__main__":
    main()
