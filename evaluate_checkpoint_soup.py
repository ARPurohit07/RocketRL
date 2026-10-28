"""Evaluate conservative interpolation between two PPO checkpoints."""

import argparse
from pathlib import Path

import torch
from stable_baselines3 import PPO

from rocket_eval import EVAL_SEEDS, evaluate_model, format_eval


TRAINED_PREFIXES = (
    "features_extractor.residual.",
    "mlp_extractor.policy_net.",
    "mlp_extractor.value_net.",
    "action_net.",
    "value_net.",
)


def pad_departures(stats):
    failures = stats["failure_categories"]
    return sum(
        failures.get(category, 0)
        for category in ("off_pad", "unsettled_contact", "out_of_bounds_lateral")
    )


def score(stats):
    return stats["success_rate"], -pad_departures(stats), stats["mean_reward"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="rocket_pythia_phase2_best.zip")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--output", type=Path, default=Path("rocket_pythia_soup_candidate.zip"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if any(not 0.0 < alpha < 1.0 for alpha in args.alphas):
        raise ValueError("Every interpolation alpha must be between zero and one")

    model = PPO.load(args.base, device=args.device)
    candidate = PPO.load(args.candidate, device=args.device)
    candidate_params = dict(candidate.policy.named_parameters())
    selected = {
        name: (parameter.detach().clone(), candidate_params[name].detach().clone())
        for name, parameter in model.policy.named_parameters()
        if name.startswith(TRAINED_PREFIXES)
    }
    print(f"Interpolating {sum(base.numel() for base, _ in selected.values()):,} trained parameters")

    baseline = evaluate_model(model, EVAL_SEEDS, challenge=1.0)
    print(format_eval("alpha=0.000 baseline", baseline), flush=True)
    best_alpha = 0.0
    best_stats = baseline

    for alpha in args.alphas:
        with torch.no_grad():
            named = dict(model.policy.named_parameters())
            for name, (base_value, candidate_value) in selected.items():
                named[name].copy_(base_value.lerp(candidate_value, alpha))
        stats = evaluate_model(model, EVAL_SEEDS, challenge=1.0)
        print(format_eval(f"alpha={alpha:.3f}", stats), flush=True)
        if score(stats) > score(best_stats):
            best_alpha = alpha
            best_stats = stats

    if best_alpha > 0.0:
        with torch.no_grad():
            named = dict(model.policy.named_parameters())
            for name, (base_value, candidate_value) in selected.items():
                named[name].copy_(base_value.lerp(candidate_value, best_alpha))
        model.save(str(args.output))
        print(
            f"Saved {args.output} alpha={best_alpha:.3f} "
            f"success={best_stats['success_rate']:.1%} "
            f"pad_departures={pad_departures(best_stats)}",
            flush=True,
        )
    else:
        print("No interpolation improved the baseline; no checkpoint saved", flush=True)


if __name__ == "__main__":
    main()
