"""Evaluate an unchanged policy under small steering-authority variations."""

import argparse

from stable_baselines3 import PPO

from rocket_eval import EVAL_SEEDS, evaluate_model, format_eval
from rocket_physics import RocketPhysics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="rocket_pythia_phase2_best.zip")
    parser.add_argument("--gimbal", nargs="+", type=float, required=True)
    parser.add_argument(
        "--torque", nargs="+", type=float,
        default=[RocketPhysics.STEERING_TORQUE],
    )
    parser.add_argument(
        "--confirmation", nargs="+", type=int,
        default=[RocketPhysics.LANDING_CONFIRMATION_STEPS],
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    model = PPO.load(args.checkpoint, device=args.device)
    original_gimbal = RocketPhysics.GIMBAL_RATIO
    original_torque = RocketPhysics.STEERING_TORQUE
    original_confirmation = RocketPhysics.LANDING_CONFIRMATION_STEPS
    try:
        for gimbal in args.gimbal:
            for torque in args.torque:
                for confirmation in args.confirmation:
                    RocketPhysics.GIMBAL_RATIO = float(gimbal)
                    RocketPhysics.STEERING_TORQUE = float(torque)
                    RocketPhysics.LANDING_CONFIRMATION_STEPS = int(confirmation)
                    stats = evaluate_model(model, EVAL_SEEDS, challenge=1.0)
                    print(
                        format_eval(
                            f"gimbal={gimbal:.3f} torque={torque:.1f} "
                            f"confirmation={confirmation}",
                            stats,
                        ),
                        flush=True,
                    )
    finally:
        RocketPhysics.GIMBAL_RATIO = original_gimbal
        RocketPhysics.STEERING_TORQUE = original_torque
        RocketPhysics.LANDING_CONFIRMATION_STEPS = original_confirmation


if __name__ == "__main__":
    main()
