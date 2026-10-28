"""Fast deterministic evaluation shared by training and reporting."""
import argparse
from collections import Counter

import numpy as np
import torch
from gymnasium.wrappers import FrameStackObservation
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

from rocket_env import RocketLandingEnv
from rocket_physics import RocketPhysics


EVAL_SEEDS = tuple(range(900_000, 900_064))


class SeedSequenceRocketEnv(RocketLandingEnv):
    """Give one vector worker a non-overlapping slice of held-out seeds."""

    def __init__(
        self, seeds, worker_index, worker_count, gamma=0.99, challenge=1.0,
        control_repeat=2,
    ):
        super().__init__(
            gamma=gamma, challenge=challenge, control_repeat=control_repeat
        )
        self.seeds = tuple(seeds)
        self.index = worker_index
        self.worker_count = worker_count

    def reset(self, *, seed=None, options=None):
        fixed_seed = self.seeds[self.index % len(self.seeds)]
        self.index += self.worker_count
        return super().reset(seed=fixed_seed, options=options)


def _failure_category(info):
    reason = info.get("crash_reason") or ""
    if "outside landing pad" in reason:
        return "off_pad"
    if "valid two-leg" in reason:
        return "unsettled_contact"
    if "Excessive descent=True" in reason:
        return "excessive_descent"
    if "Excessive horizontal speed=True" in reason:
        return "excessive_horizontal_speed"
    if "Excessive tilt=True" in reason or "Excessive rotation=True" in reason:
        return "attitude"
    if info.get("status") == "out_of_bounds":
        if "lateral" in reason.lower():
            return "out_of_bounds_lateral"
        if "altitude" in reason.lower():
            return "out_of_bounds_altitude"
        return "out_of_bounds"
    if info.get("status") == "timeout":
        return "timeout"
    return "other"


def _height_band(height):
    if height < 8.0:
        return "04-08m"
    if height < 16.0:
        return "08-16m"
    if height < 22.0:
        return "16-22m"
    return "22-28m"


def mirror_observations(observations):
    """Mirror stacked rocket states across the pad centerline."""
    mirrored = np.array(observations, copy=True)
    mirrored[..., [0, 2, 4, 5]] *= -1.0
    left_contact = mirrored[..., 6].copy()
    mirrored[..., 6] = mirrored[..., 7]
    mirrored[..., 7] = left_contact
    return mirrored


def unmirror_action_probabilities(probabilities):
    """Map actions from mirrored coordinates back to world coordinates."""
    return np.asarray(probabilities)[:, [0, 3, 2, 1]]


def initial_canonical_mask(
    observations, canonical_side="positive", x_threshold=0.0,
    min_height=0.0, max_outward_tilt=99.0,
    emergency_height=10.0, emergency_x=2.0, emergency_x_max=4.0,
):
    """Select starts where transferring the stronger side is recoverable."""
    if canonical_side not in ("positive", "negative"):
        raise ValueError("canonical_side must be 'positive' or 'negative'")
    latest = np.asarray(observations)[:, -1]
    x_m = latest[:, 0] * RocketPhysics.X_OBSERVATION_SCALE
    height_m = latest[:, 1] * RocketPhysics.ALTITUDE_OBSERVATION_SCALE + 1.5
    vx_mps = latest[:, 2] * 10.0
    angle = latest[:, 4]
    enough_height = height_m >= float(min_height)
    if canonical_side == "positive":
        displaced = x_m < -float(x_threshold)
        safe_tilt = angle <= float(max_outward_tilt)
        low_outward_emergency = (
            (height_m < float(emergency_height))
            & (x_m < -float(emergency_x))
            & (x_m >= -float(emergency_x_max))
            & (vx_mps < 0.0)
            & (angle > 0.0)
        )
    else:
        displaced = x_m > float(x_threshold)
        safe_tilt = angle >= -float(max_outward_tilt)
        low_outward_emergency = (
            (height_m < float(emergency_height))
            & (x_m > float(emergency_x))
            & (x_m <= float(emergency_x_max))
            & (vx_mps > 0.0)
            & (angle < 0.0)
        )
    return displaced & enough_height & safe_tilt & ~low_outward_emergency


def policy_probabilities(
    model, observations, symmetry_weight=0.0, canonical_side=None,
    canonical_threshold=0.0, canonical_mask=None,
):
    """Return policy probabilities, optionally averaged with mirrored states."""
    symmetry_weight = float(symmetry_weight)
    if not 0.0 <= symmetry_weight <= 1.0:
        raise ValueError("symmetry_weight must be between zero and one")
    if canonical_mask is not None:
        mask = np.asarray(canonical_mask, dtype=bool)
        if np.all(mask) or not np.any(mask):
            selected = mirror_observations(observations) if np.all(mask) else observations
            selected_tensor, _ = model.policy.obs_to_tensor(selected)
            selected_probabilities = (
                model.policy.get_distribution(selected_tensor).distribution.probs
                .detach().cpu().numpy()
            )
            return (
                unmirror_action_probabilities(selected_probabilities)
                if np.all(mask) else selected_probabilities
            )
    if symmetry_weight == 0.0 and canonical_side is None and canonical_mask is None:
        obs_tensor, _ = model.policy.obs_to_tensor(observations)
        return (
            model.policy.get_distribution(obs_tensor).distribution.probs
            .detach().cpu().numpy()
        )

    mirrored = mirror_observations(observations)
    combined = np.concatenate([observations, mirrored], axis=0)
    combined_tensor, _ = model.policy.obs_to_tensor(combined)
    combined_probabilities = (
        model.policy.get_distribution(combined_tensor).distribution.probs
        .detach().cpu().numpy()
    )
    probabilities, mirrored_probabilities = np.split(combined_probabilities, 2)
    # Map the mirrored policy back to real-world actions: 1 <-> 3.
    mirrored_probabilities = unmirror_action_probabilities(mirrored_probabilities)
    if canonical_mask is not None:
        return np.where(
            np.asarray(canonical_mask, dtype=bool)[:, None],
            mirrored_probabilities,
            probabilities,
        )
    if canonical_side is not None:
        if canonical_side not in ("positive", "negative"):
            raise ValueError("canonical_side must be 'positive', 'negative', or None")
        latest_x = np.asarray(observations)[:, -1, 0]
        normalized_threshold = float(canonical_threshold) / 15.0
        use_mirrored = (
            latest_x < -normalized_threshold
            if canonical_side == "positive"
            else latest_x > normalized_threshold
        )
        return np.where(use_mirrored[:, None], mirrored_probabilities, probabilities)
    return (
        (1.0 - symmetry_weight) * probabilities
        + symmetry_weight * mirrored_probabilities
    )


def evaluate_model(
    model, seeds=EVAL_SEEDS, num_envs=8, gamma=0.99, challenge=1.0,
    deterministic=True, sampling_seed=12345, control_repeat=2,
    symmetry_weight=0.0, canonical_side=None, canonical_threshold=0.0,
    canonical_initial_side=None, canonical_initial_x_threshold=0.0,
    canonical_initial_min_height=0.0,
    canonical_initial_max_outward_tilt=99.0,
    canonical_initial_emergency_height=10.0,
    canonical_initial_emergency_x=2.0,
    canonical_initial_emergency_x_max=4.0,
):
    """Evaluate each seed once while batching policy inference across envs."""
    seeds = tuple(seeds)
    worker_count = min(num_envs, len(seeds))

    def make_env(worker_index):
        def factory():
            env = SeedSequenceRocketEnv(
                seeds, worker_index, worker_count, gamma=gamma,
                challenge=challenge, control_repeat=control_repeat,
            )
            return Monitor(
                FrameStackObservation(env, 8),
                info_keywords=("is_success",),
            )
        return factory

    env = DummyVecEnv([make_env(i) for i in range(worker_count)])
    quotas = np.array(
        [len(range(i, len(seeds), worker_count)) for i in range(worker_count)],
        dtype=np.int32,
    )
    completed = np.zeros(worker_count, dtype=np.int32)
    totals = np.zeros(worker_count, dtype=np.float64)
    episode_actions = [Counter() for _ in range(worker_count)]
    rewards, successes, episode_records = [], [], []
    statuses, failures, actions = Counter(), Counter(), Counter()
    probability_sum = np.zeros(4, dtype=np.float64)
    entropy_sum = 0.0
    probability_states = 0
    bands = {}
    sampling_rng = np.random.default_rng(sampling_seed)

    try:
        obs = env.reset()
        canonical_mask = None
        if canonical_initial_side is not None:
            if canonical_initial_side not in ("positive", "negative"):
                raise ValueError(
                    "canonical_initial_side must be 'positive', 'negative', or None"
                )
            canonical_mask = initial_canonical_mask(
                obs, canonical_side=canonical_initial_side,
                x_threshold=canonical_initial_x_threshold,
                min_height=canonical_initial_min_height,
                max_outward_tilt=canonical_initial_max_outward_tilt,
                emergency_height=canonical_initial_emergency_height,
                emergency_x=canonical_initial_emergency_x,
                emergency_x_max=canonical_initial_emergency_x_max,
            )
        while np.any(completed < quotas):
            with torch.no_grad():
                probabilities = policy_probabilities(
                    model, obs, symmetry_weight=symmetry_weight,
                    canonical_side=canonical_side,
                    canonical_threshold=canonical_threshold,
                    canonical_mask=canonical_mask,
                )
                if deterministic:
                    selected = np.argmax(probabilities, axis=1)
                else:
                    selected = np.asarray([
                        sampling_rng.choice(4, p=probability / probability.sum())
                        for probability in probabilities
                    ])
            for i, action in enumerate(selected):
                if completed[i] < quotas[i]:
                    action = int(action)
                    actions[action] += 1
                    episode_actions[i][action] += 1
                    probability_sum += probabilities[i]
                    entropy_sum += float(
                        -np.sum(probabilities[i] * np.log(probabilities[i] + 1e-12))
                    )
                    probability_states += 1
            obs, reward, dones, infos = env.step(selected)
            totals += np.asarray(reward, dtype=np.float64)

            for i, done in enumerate(dones):
                if not done:
                    continue
                if completed[i] < quotas[i]:
                    info = infos[i]
                    episode_index = i + int(completed[i]) * worker_count
                    episode_seed = int(seeds[episode_index])
                    status = info.get("status", "unknown")
                    success = bool(info.get("is_success", False))
                    rewards.append(float(totals[i]))
                    successes.append(success)
                    statuses[status] += 1
                    failure = None if success else _failure_category(info)
                    if not success:
                        failures[failure] += 1
                    episode_records.append({
                        "seed": episode_seed,
                        "reward": float(totals[i]),
                        "success": success,
                        "status": status,
                        "failure": failure,
                        "start": dict(info.get("start", {})),
                        "final_x": float(info.get("x", 0.0)),
                        "final_vx": float(info.get("vx", 0.0)),
                        "final_vy": float(info.get("vy", 0.0)),
                        "final_angle": float(info.get("angle", 0.0)),
                        "final_omega": float(info.get("angular_velocity", 0.0)),
                    })
                    height = float(info.get("start", {}).get("y", 0.0))
                    band = _height_band(height)
                    band_stats = bands.setdefault(band, {"episodes": 0, "successes": 0})
                    band_stats["episodes"] += 1
                    band_stats["successes"] += int(success)
                    completed[i] += 1
                totals[i] = 0.0
                episode_actions[i].clear()
                if canonical_mask is not None:
                    canonical_mask[i] = initial_canonical_mask(
                        obs[i:i + 1], canonical_side=canonical_initial_side,
                        x_threshold=canonical_initial_x_threshold,
                        min_height=canonical_initial_min_height,
                        max_outward_tilt=canonical_initial_max_outward_tilt,
                        emergency_height=canonical_initial_emergency_height,
                        emergency_x=canonical_initial_emergency_x,
                        emergency_x_max=canonical_initial_emergency_x_max,
                    )[0]
    finally:
        env.close()

    by_height = {
        band: {
            **values,
            "success_rate": values["successes"] / values["episodes"],
        }
        for band, values in sorted(bands.items())
    }
    return {
        "episodes": len(rewards),
        "success_rate": float(np.mean(successes)),
        "mean_reward": float(np.mean(rewards)),
        "max_reward": float(np.max(rewards)),
        "statuses": dict(statuses),
        "failure_categories": dict(failures),
        "action_counts": dict(sorted(actions.items())),
        "mean_action_probabilities": (probability_sum / probability_states).tolist(),
        "mean_entropy": entropy_sum / probability_states,
        "by_height": by_height,
        "episode_records": sorted(episode_records, key=lambda row: row["seed"]),
    }


def format_eval(prefix, stats):
    height_summary = ", ".join(
        f"{band}:{values['success_rate']:.0%}"
        for band, values in stats["by_height"].items()
    )
    return (
        f"{prefix}: reward={stats['mean_reward']:.2f} "
        f"success={stats['success_rate']:.1%} max={stats['max_reward']:.2f} "
        f"heights=[{height_summary}] actions={stats['action_counts']} "
        f"probs={[round(value, 3) for value in stats['mean_action_probabilities']]} "
        f"entropy={stats['mean_entropy']:.3f} "
        f"failures={stats['failure_categories']}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="rocket_pythia_phase2_best.zip")
    parser.add_argument("--episodes", type=int, default=len(EVAL_SEEDS))
    parser.add_argument("--seed-start", type=int, default=EVAL_SEEDS[0])
    parser.add_argument("--challenge", type=float, default=1.0)
    parser.add_argument("--control-repeat", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--symmetric-policy", action="store_true")
    parser.add_argument("--symmetry-weight", type=float, default=0.0)
    parser.add_argument("--canonical-side", choices=("positive", "negative"))
    parser.add_argument("--canonical-threshold", type=float, default=0.0)
    parser.add_argument(
        "--canonical-initial-side", choices=("positive", "negative")
    )
    parser.add_argument("--canonical-initial-x-threshold", type=float, default=0.0)
    parser.add_argument("--canonical-initial-min-height", type=float, default=0.0)
    parser.add_argument(
        "--canonical-initial-max-outward-tilt", type=float, default=99.0
    )
    parser.add_argument(
        "--canonical-initial-emergency-height", type=float, default=10.0
    )
    parser.add_argument(
        "--canonical-initial-emergency-x", type=float, default=2.0
    )
    parser.add_argument(
        "--canonical-initial-emergency-x-max", type=float, default=4.0
    )
    parser.add_argument(
        "--raw-policy", action="store_true",
        help="evaluate the checkpoint without deployment symmetry handling",
    )
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")
    seeds = tuple(range(args.seed_start, args.seed_start + args.episodes))
    model = PPO.load(args.checkpoint, device=args.device)
    stats = evaluate_model(
        model,
        seeds=seeds,
        num_envs=min(8, args.episodes),
        challenge=args.challenge,
        deterministic=not args.stochastic,
        control_repeat=args.control_repeat,
        symmetry_weight=(0.5 if args.symmetric_policy else args.symmetry_weight),
        canonical_side=args.canonical_side,
        canonical_threshold=args.canonical_threshold,
        canonical_initial_side=(
            None if args.raw_policy
            else (args.canonical_initial_side or "positive")
        ),
        canonical_initial_x_threshold=args.canonical_initial_x_threshold,
        canonical_initial_min_height=args.canonical_initial_min_height,
        canonical_initial_max_outward_tilt=(
            args.canonical_initial_max_outward_tilt
        ),
        canonical_initial_emergency_height=(
            args.canonical_initial_emergency_height
        ),
        canonical_initial_emergency_x=args.canonical_initial_emergency_x,
        canonical_initial_emergency_x_max=args.canonical_initial_emergency_x_max,
    )
    print(format_eval("Evaluation", stats))


if __name__ == "__main__":
    main()
