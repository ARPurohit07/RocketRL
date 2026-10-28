"""Sweep selective mirrored-policy transfer thresholds."""

import argparse

from stable_baselines3 import PPO

from rocket_eval import EVAL_SEEDS, evaluate_model, format_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="rocket_pythia_phase2_best.zip")
    parser.add_argument("--side", choices=("positive", "negative"), default="positive")
    parser.add_argument("--thresholds", nargs="+", type=float, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    model = PPO.load(args.checkpoint, device=args.device)
    for threshold in args.thresholds:
        stats = evaluate_model(
            model, EVAL_SEEDS, challenge=1.0,
            canonical_side=args.side, canonical_threshold=threshold,
        )
        print(
            format_eval(
                f"canonical={args.side} threshold={threshold:.2f}m", stats
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
