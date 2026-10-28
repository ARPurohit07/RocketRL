"""Sweep mirrored-policy blending without changing checkpoint weights."""

import argparse

from stable_baselines3 import PPO

from rocket_eval import EVAL_SEEDS, evaluate_model, format_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="rocket_pythia_phase2_best.zip")
    parser.add_argument("--weights", nargs="+", type=float, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    model = PPO.load(args.checkpoint, device=args.device)
    for weight in args.weights:
        stats = evaluate_model(
            model, EVAL_SEEDS, challenge=1.0, symmetry_weight=weight
        )
        print(format_eval(f"symmetry_weight={weight:.3f}", stats), flush=True)


if __name__ == "__main__":
    main()
