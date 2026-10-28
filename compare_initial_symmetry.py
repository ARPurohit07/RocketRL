"""Compare raw and mirrored deployment per seed and score simple safe gates."""

import argparse
from collections import Counter

import numpy as np
from stable_baselines3 import PPO

from rocket_eval import evaluate_model, format_eval


def summarize_selected(rows, label):
    rewards = [row["reward"] for row in rows]
    successes = [row["success"] for row in rows]
    failures = Counter(row["failure"] for row in rows if not row["success"])
    pad_departures = sum(
        failures[name]
        for name in ("off_pad", "unsettled_contact", "out_of_bounds_lateral")
    )
    print(
        f"{label}: reward={np.mean(rewards):+.2f} "
        f"success={np.mean(successes):.1%} pad_departures={pad_departures} "
        f"failures={dict(failures)}",
        flush=True,
    )


def metrics(rows):
    failures = Counter(row["failure"] for row in rows if not row["success"])
    return {
        "successes": sum(row["success"] for row in rows),
        "reward": float(np.mean([row["reward"] for row in rows])),
        "lateral": failures["out_of_bounds_lateral"],
        "pad_departures": sum(
            failures[name]
            for name in ("off_pad", "unsettled_contact", "out_of_bounds_lateral")
        ),
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="rocket_pythia_phase2_best.zip")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=930_000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    seeds = tuple(range(args.seed_start, args.seed_start + args.episodes))
    model = PPO.load(args.checkpoint, device=args.device)
    raw = evaluate_model(model, seeds=seeds, canonical_initial_side=None)
    mirrored = evaluate_model(
        model, seeds=seeds, canonical_initial_side="positive",
        canonical_initial_x_threshold=0.0,
        canonical_initial_min_height=0.0,
        canonical_initial_max_outward_tilt=99.0,
        canonical_initial_emergency_height=0.0,
    )
    print(format_eval("raw", raw), flush=True)
    print(format_eval("mirror-negative", mirrored), flush=True)

    raw_rows = {row["seed"]: row for row in raw["episode_records"]}
    mirrored_rows = {row["seed"]: row for row in mirrored["episode_records"]}
    for threshold in np.arange(0.0, 8.01, 0.5):
        selected = []
        for seed in seeds:
            raw_row = raw_rows[seed]
            use_mirror = float(raw_row["start"]["x"]) < -float(threshold)
            selected.append(mirrored_rows[seed] if use_mirror else raw_row)
        summarize_selected(selected, f"initial_x < -{threshold:.1f}m")

    candidates = []
    for x_threshold in np.arange(0.0, 2.01, 0.5):
        for min_height in (0.0, 6.0, 8.0, 10.0):
            for max_outward_tilt in (99.0, 0.30, 0.25, 0.22, 0.20):
                selected = []
                for seed in seeds:
                    raw_row = raw_rows[seed]
                    start = raw_row["start"]
                    use_mirror = (
                        float(start["x"]) < -x_threshold
                        and float(start["y"]) >= min_height
                        and float(start["angle"]) <= max_outward_tilt
                    )
                    selected.append(mirrored_rows[seed] if use_mirror else raw_row)
                candidates.append((
                    metrics(selected), x_threshold, min_height,
                    max_outward_tilt,
                ))
    candidates.sort(key=lambda item: (
        -item[0]["successes"], item[0]["lateral"],
        item[0]["pad_departures"], -item[0]["reward"],
    ))
    print("\nBest initial-state gates:", flush=True)
    for result, x_threshold, min_height, max_tilt in candidates[:12]:
        print(
            f"  success={result['successes']}/{len(seeds)} "
            f"lateral={result['lateral']} pad_departures={result['pad_departures']} "
            f"reward={result['reward']:+.2f} x>{x_threshold:.1f}m "
            f"height>={min_height:.0f}m outward_tilt<={max_tilt:.2f}",
            flush=True,
        )

    print("\nChanged outcomes for negative-X starts:", flush=True)
    transitions = Counter()
    for seed in seeds:
        raw_row, mirrored_row = raw_rows[seed], mirrored_rows[seed]
        if float(raw_row["start"]["x"]) >= 0.0:
            continue
        raw_outcome = "landed" if raw_row["success"] else raw_row["failure"]
        mirrored_outcome = (
            "landed" if mirrored_row["success"] else mirrored_row["failure"]
        )
        transitions[(raw_outcome, mirrored_outcome)] += 1
    for transition, count in transitions.most_common():
        print(f"  {transition[0]} -> {transition[1]}: {count}", flush=True)

    print("\nNegative-X starts where mirroring changed success:", flush=True)
    for seed in seeds:
        raw_row, mirrored_row = raw_rows[seed], mirrored_rows[seed]
        start = raw_row["start"]
        if float(start["x"]) >= 0.0 or raw_row["success"] == mirrored_row["success"]:
            continue
        raw_outcome = "landed" if raw_row["success"] else raw_row["failure"]
        mirrored_outcome = (
            "landed" if mirrored_row["success"] else mirrored_row["failure"]
        )
        print(
            f"  seed={seed} {raw_outcome}->{mirrored_outcome} "
            f"x={start['x']:+.2f} y={start['y']:.2f} "
            f"vx={start['vx']:+.2f} vy={start['vy']:+.2f} "
            f"angle={start['angle']:+.3f} "
            f"omega={start['angular_velocity']:+.3f}",
            flush=True,
        )

    print("\nMirrored boundary/attitude regressions:", flush=True)
    for seed in seeds:
        raw_row, mirrored_row = raw_rows[seed], mirrored_rows[seed]
        start = raw_row["start"]
        if (
            float(start["x"]) >= 0.0
            or mirrored_row["failure"] not in ("out_of_bounds_lateral", "attitude")
            or raw_row["failure"] == mirrored_row["failure"]
        ):
            continue
        raw_outcome = "landed" if raw_row["success"] else raw_row["failure"]
        print(
            f"  seed={seed} {raw_outcome}->{mirrored_row['failure']} "
            f"x={start['x']:+.2f} y={start['y']:.2f} "
            f"vx={start['vx']:+.2f} vy={start['vy']:+.2f} "
            f"angle={start['angle']:+.3f} "
            f"omega={start['angular_velocity']:+.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
