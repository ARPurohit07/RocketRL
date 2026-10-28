"""Measure deterministic policy robustness across continuous challenge levels."""
import argparse

from stable_baselines3 import PPO

from rocket_eval import evaluate_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="rocket_pythia_phase2_best.zip")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--control-repeat", type=int, default=2)
    parser.add_argument(
        "--challenges", type=float, nargs="+",
        default=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0),
    )
    args = parser.parse_args()

    model = PPO.load(args.model, device="cpu")
    seeds = tuple(range(900_000, 900_000 + args.episodes))
    for challenge in args.challenges:
        stats = evaluate_model(
            model,
            seeds=seeds,
            num_envs=min(8, args.episodes),
            challenge=challenge,
            control_repeat=args.control_repeat,
            canonical_initial_side="positive",
        )
        heights = ",".join(
            f"{band}:{values['success_rate']:.0%}"
            for band, values in stats["by_height"].items()
        )
        print(
            f"FRONTIER challenge={challenge:.2f} "
            f"success={stats['success_rate']:.1%} "
            f"mean={stats['mean_reward']:+.2f} max={stats['max_reward']:+.2f} "
            f"heights={heights} failures={stats['failure_categories']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
