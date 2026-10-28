"""Evaluate a Pythia PPO checkpoint on fixed, unseen randomized scenarios."""
import argparse, json
from collections import Counter
from pathlib import Path
import numpy as np
from gymnasium.wrappers import FrameStackObservation
from stable_baselines3 import PPO
from rocket_env import RocketLandingEnv

SEEDS = list(range(900_000, 900_100))

def evaluate(checkpoint, episodes=100, device="cuda"):
    env = FrameStackObservation(RocketLandingEnv(), 8)
    model = PPO.load(str(checkpoint), env=env, device=device)
    rewards, touchdown_vx, touchdown_vy, errors, fuel, times = [], [], [], [], [], []
    statuses, reasons, categories = Counter(), Counter(), Counter()
    actions = Counter()
    actions_by_status = {}
    rewards_by_status = {}
    for i, seed in enumerate(SEEDS[:episodes]):
        obs, info = env.reset(seed=seed)
        total = 0.0
        episode_actions = Counter()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            actions[action] += 1; episode_actions[action] += 1
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            if terminated or truncated:
                break
        status = info["status"]
        rewards.append(total); statuses[status] += 1
        rewards_by_status.setdefault(status, []).append(total)
        status_actions = actions_by_status.setdefault(status, Counter())
        status_actions.update(episode_actions)
        reason = info.get("crash_reason")
        if reason:
            reasons[reason] += 1
            if "outside landing pad" in reason:
                categories["off_pad"] += 1
            elif "valid two-leg" in reason:
                categories["unsettled_contact"] += 1
            elif "Excessive descent=True" in reason:
                categories["excessive_descent"] += 1
            elif "Excessive horizontal speed=True" in reason:
                categories["excessive_horizontal_speed"] += 1
            elif "Excessive tilt=True" in reason:
                categories["excessive_tilt"] += 1
            elif "time limit" in reason:
                categories["timeout"] += 1
            else:
                categories["other_crash"] += 1
        elif info["status"] == "out_of_bounds":
            categories["out_of_bounds"] += 1
        if info["status"] == "landed":
            touchdown_vx.append(info["touchdown_vx"]); touchdown_vy.append(info["touchdown_vy"])
            errors.append(info["landing_error"]); fuel.append(info["fuel"]); times.append(info["landing_time"])
    env.close()
    return {
        "checkpoint": str(checkpoint), "randomized_starts": True, "episodes": episodes,
        "success_rate": statuses["landed"] / episodes, "crash_rate": statuses["crashed"] / episodes,
        "out_of_bounds_rate": statuses["out_of_bounds"] / episodes,
        "mean_reward": float(np.mean(rewards)), "max_reward": float(np.max(rewards)),
        "mean_reward_by_status": {
            status: float(np.mean(values)) for status, values in rewards_by_status.items()
        },
        "action_counts": dict(sorted(actions.items())),
        "action_counts_by_status": {
            status: dict(sorted(counts.items())) for status, counts in actions_by_status.items()
        },
        "mean_touchdown_vx": float(np.mean(touchdown_vx)) if touchdown_vx else None,
        "max_touchdown_vx": float(np.max(touchdown_vx)) if touchdown_vx else None,
        "mean_touchdown_vy": float(np.mean(touchdown_vy)) if touchdown_vy else None,
        "max_touchdown_vy": float(np.max(touchdown_vy)) if touchdown_vy else None,
        "mean_landing_error": float(np.mean(errors)) if errors else None,
        "mean_fuel_remaining": float(np.mean(fuel)) if fuel else None,
        "mean_landing_time": float(np.mean(times)) if times else None,
        "failure_categories": categories, "failure_reasons": reasons,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="backups/phase2_baseline_best_90pct.zip")
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    results = [evaluate(args.checkpoint, args.episodes)]
    out = Path(f"evaluation_randomized_{Path(args.checkpoint).stem}.json")
    out.write_text(json.dumps(results, indent=2, default=lambda x: dict(x)) + "\n")
    for result in results:
        print(json.dumps(result, indent=2, default=lambda x: dict(x)))
    print("Saved", out)

if __name__ == "__main__":
    main()
