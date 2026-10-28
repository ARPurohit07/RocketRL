"""Measure deterministic and sampled behavior as action logits are sharpened."""
import argparse
from pathlib import Path

import torch
from stable_baselines3 import PPO

from rocket_eval import EVAL_SEEDS, evaluate_model, format_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("rocket_pythia_phase2_best.zip"))
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--challenge", type=float, default=0.0)
    parser.add_argument("--scales", type=float, nargs="+", default=(1.0, 2.0, 4.0, 8.0))
    args = parser.parse_args()

    model = PPO.load(str(args.checkpoint), device="cuda")
    seeds = EVAL_SEEDS[:args.episodes]
    layer = model.policy.action_net
    original_weight = layer.weight.detach().clone()
    original_bias = layer.bias.detach().clone()
    try:
        deterministic = evaluate_model(
            model, seeds, challenge=args.challenge, deterministic=True
        )
        print(format_eval("Deterministic reference", deterministic), flush=True)
        for scale in args.scales:
            with torch.no_grad():
                layer.weight.copy_(original_weight * scale)
                layer.bias.copy_(original_bias * scale)
            sampled = evaluate_model(
                model, seeds, challenge=args.challenge, deterministic=False
            )
            print(format_eval(f"Sampled logit scale {scale:g}", sampled), flush=True)
    finally:
        with torch.no_grad():
            layer.weight.copy_(original_weight)
            layer.bias.copy_(original_bias)


if __name__ == "__main__":
    main()
