"""PPO fine-tuning on recoverable randomized starts."""
import argparse, gc, json
from collections import deque
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from gymnasium.wrappers import FrameStackObservation
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from pythia_controller import PythiaController
from rocket_env import RocketLandingEnv
from rocket_eval import EVAL_SEEDS, evaluate_model, format_eval


class BroadEvalCallback(BaseCallback):
    """Evaluate exact checkpoints on broad and core held-out suites."""

    def __init__(self, eval_freq, run_dir, eval_envs=8, control_repeat=2):
        super().__init__(verbose=0)
        self.eval_freq = eval_freq
        self.run_dir = Path(run_dir)
        self.eval_envs = eval_envs
        self.control_repeat = control_repeat
        self.records = []

    def _on_step(self):
        if self.n_calls % self.eval_freq:
            return True
        stats = evaluate_model(
            self.model, EVAL_SEEDS, num_envs=self.eval_envs, challenge=1.0,
            control_repeat=self.control_repeat,
            canonical_initial_side="positive",
        )
        core_stats = evaluate_model(
            self.model, EVAL_SEEDS, num_envs=self.eval_envs, challenge=0.0,
            control_repeat=self.control_repeat,
            canonical_initial_side="positive",
        )
        timestep = int(self.num_timesteps)
        checkpoint = self.run_dir / f"evaluated_{timestep}_steps.zip"
        self.model.save(str(checkpoint))
        record = {
            "timestep": timestep,
            "checkpoint": str(checkpoint),
            **stats,
            "core_success_rate": core_stats["success_rate"],
            "core_mean_reward": core_stats["mean_reward"],
        }
        self.records.append(record)
        with (self.run_dir / "evaluations.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(format_eval(f"Held-out evaluation at {timestep} steps", stats), flush=True)
        print(format_eval(f"Core retention at {timestep} steps", core_stats), flush=True)
        return True


class ContinuousChallengeCallback(BaseCallback):
    """Smoothly widen one randomized distribution during a training run."""

    def __init__(self, start, end, run_steps, base_timesteps):
        super().__init__(verbose=0)
        self.start = start
        self.end = end
        self.run_steps = max(1, run_steps)
        self.base_timesteps = base_timesteps
        self.last_bucket = -1

    def _on_step(self):
        progress = float(np.clip(
            (self.num_timesteps - self.base_timesteps) / self.run_steps, 0.0, 1.0
        ))
        challenge = self.start + progress * (self.end - self.start)
        for wrapped in self.training_env.envs:
            wrapped.unwrapped.set_challenge(challenge)
        bucket = int(progress * 10)
        if bucket != self.last_bucket:
            self.last_bucket = bucket
            print(
                f"Randomization challenge={challenge:.3f} "
                f"progress={progress:.0%}",
                flush=True,
            )
        return True


class AdaptiveChallengeCallback(BaseCallback):
    """Keep randomized starts near the policy's current learning frontier."""

    def __init__(
        self, start, end, target_success=0.55, window_episodes=128,
        step_up=0.025, step_down=0.0125,
    ):
        super().__init__(verbose=0)
        self.minimum = float(start)
        self.maximum = float(end)
        self.challenge = float(start)
        self.target_success = float(target_success)
        self.window_episodes = int(window_episodes)
        self.step_up = float(step_up)
        self.step_down = float(step_down)
        self.outcomes = deque(maxlen=self.window_episodes)

    def _set_challenge(self):
        for wrapped in self.training_env.envs:
            wrapped.unwrapped.set_challenge(self.challenge)

    def _on_training_start(self):
        self._set_challenge()
        print(
            f"Adaptive curriculum challenge={self.challenge:.3f} "
            f"range=[{self.minimum:.3f}, {self.maximum:.3f}] "
            f"target_success={self.target_success:.0%}",
            flush=True,
        )

    def _on_step(self):
        dones = self.locals.get("dones", ())
        infos = self.locals.get("infos", ())
        for done, info in zip(dones, infos):
            if done:
                self.outcomes.append(float(bool(info.get("is_success", False))))
        if len(self.outcomes) < self.window_episodes:
            return True

        success_rate = float(np.mean(self.outcomes))
        previous = self.challenge
        if success_rate >= self.target_success:
            self.challenge = min(self.maximum, self.challenge + self.step_up)
        elif success_rate < self.target_success - 0.12:
            self.challenge = max(self.minimum, self.challenge - self.step_down)
        self.outcomes.clear()
        self._set_challenge()
        direction = "hold" if self.challenge == previous else (
            "raise" if self.challenge > previous else "lower"
        )
        print(
            f"Adaptive curriculum {direction}: success={success_rate:.1%} "
            f"challenge={self.challenge:.3f}",
            flush=True,
        )
        return True


class EntropyScheduleCallback(BaseCallback):
    """Explore steering early, then let PPO consolidate decisive actions."""

    def __init__(self, start, end, run_steps, base_timesteps):
        super().__init__(verbose=0)
        self.start = start
        self.end = end
        self.run_steps = max(1, run_steps)
        self.base_timesteps = base_timesteps
        self.last_bucket = -1

    def _on_step(self):
        progress = float(np.clip(
            (self.num_timesteps - self.base_timesteps) / self.run_steps, 0.0, 1.0
        ))
        self.model.ent_coef = self.start + progress * (self.end - self.start)
        bucket = int(progress * 10)
        if bucket != self.last_bucket:
            self.last_bucket = bucket
            print(
                f"Entropy coefficient={self.model.ent_coef:.4f} "
                f"progress={progress:.0%}",
                flush=True,
            )
        return True


def _pad_departure_failures(stats):
    failures = stats.get("failure_categories", {})
    return sum(
        failures.get(category, 0)
        for category in ("off_pad", "unsettled_contact", "out_of_bounds_lateral")
    )


def _lateral_boundary_failures(stats):
    return stats.get("failure_categories", {}).get("out_of_bounds_lateral", 0)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path("rocket_pythia_phase2_best.zip"))
    p.add_argument("--resume", action="store_true", help="continue from the promoted best checkpoint")
    p.add_argument(
        "--fresh", action="store_true",
        help="initialize a new PPO policy with the same Pythia-70M + LoRA architecture",
    )
    p.add_argument("--steps", type=int, default=20480); p.add_argument("--smoke", action="store_true")
    p.add_argument(
        "--eval-interval", type=int, default=8192,
        help="training timesteps between exact held-out checkpoint evaluations",
    )
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--control-repeat", type=int, default=2)
    p.add_argument("--challenge-start", type=float, default=0.0)
    p.add_argument("--challenge-end", type=float, default=0.45)
    p.add_argument(
        "--adaptive-curriculum", action="store_true",
        help="adjust challenge from recent reward-driven episode success",
    )
    p.add_argument("--curriculum-target", type=float, default=0.55)
    p.add_argument("--curriculum-window", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--antithetic-starts", action="store_true",
        help="pair every random training start with its left/right mirror",
    )
    p.add_argument("--entropy-start", type=float, default=0.05)
    p.add_argument("--entropy-end", type=float, default=0.0)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--n-epochs", type=int, default=6)
    p.add_argument("--clip-range", type=float, default=0.15)
    p.add_argument("--vf-coef", type=float, default=0.25)
    p.add_argument(
        "--logit-scale", type=float, default=2.0,
        help="sharpen a legacy checkpoint before a new stage; preserves argmax actions",
    )
    p.add_argument(
        "--side-logit-bias", type=float, default=0.0,
        help="equal exploration offset for both steering actions on a new stage",
    )
    args = p.parse_args()
    if args.eval_interval < 1:
        raise ValueError("--eval-interval must be at least 1")
    model_path = Path("rocket_pythia_phase2_latest.zip")
    best_model_path = Path("rocket_pythia_phase2_best.zip")
    config = {"physics_version": 14, "reward_version": 15, "randomized_starts": "continuous_uniform_v2",
              "observation_shape": [8, 9], "stack_size": 8, "gamma": .99,
              "num_envs": args.num_envs,
              "control_repeat": args.control_repeat,
              "n_steps": 256, "batch_size": 256, "n_epochs": args.n_epochs,
              "learning_rate": args.learning_rate,
              "clip_range": args.clip_range, "target_kl": .02,
              "vf_coef": args.vf_coef,
              "trainable_scope": (
                  "lora_adapters_and_ppo" if args.fresh
                  else "ppo_heads_and_existing_residual"
              ),
              "fresh_initialization": args.fresh,
              "initial_logit_scale": args.logit_scale,
              "initial_side_logit_bias": args.side_logit_bias,
              "entropy_start": args.entropy_start, "entropy_end": args.entropy_end,
              "challenge_start": args.challenge_start, "challenge_end": args.challenge_end,
              "adaptive_curriculum": args.adaptive_curriculum,
              "curriculum_target": args.curriculum_target,
              "curriculum_window": args.curriculum_window,
              "antithetic_starts": args.antithetic_starts,
              "eval_interval": args.eval_interval,
              "eval_seeds": len(EVAL_SEEDS), "eval_envs": 8,
              "inference_policy": "gated_canonical_positive_initial_side",
              "canonical_initial_x_threshold": 0.0,
              "canonical_initial_min_height": 0.0,
              "canonical_initial_max_outward_tilt": 99.0,
              "canonical_initial_emergency_height": 10.0,
              "canonical_initial_emergency_x": 2.0,
              "canonical_initial_emergency_x_max": 4.0,
              "reset_optimizer_on_new_stage": True, "seed": args.seed}
    config_path = Path("rocket_pythia_phase2_config.json")
    def make_env():
        return Monitor(
            FrameStackObservation(
                RocketLandingEnv(
                    gamma=.99, challenge=args.challenge_start,
                    control_repeat=args.control_repeat,
                    antithetic_starts=args.antithetic_starts,
                ), 8
            ),
            info_keywords=("is_success",),
        )
    torch.set_num_threads(2)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU not detected")
    n_envs, steps = (1, 256) if args.smoke else (args.num_envs, args.steps)
    # Box2D stepping is cheap; an in-process vector avoids importing the full
    # PyTorch/CUDA stack once per Windows spawn worker.
    env = DummyVecEnv([make_env for _ in range(n_envs)])
    env.seed(args.seed)
    try:
        resume = args.resume
        if resume and args.fresh:
            raise ValueError("--resume and --fresh are mutually exclusive")
        if resume and not (
            best_model_path.exists()
            and config_path.exists()
            and json.loads(config_path.read_text()) == config
        ):
            raise RuntimeError("Cannot resume: promoted best checkpoint/config is missing or incompatible")
        if args.fresh:
            model = PPO(
                "MlpPolicy", env,
                policy_kwargs={
                    "features_extractor_class": PythiaController,
                    "net_arch": {"pi": [64], "vf": [64]},
                },
                learning_rate=config["learning_rate"],
                n_steps=config["n_steps"], batch_size=config["batch_size"],
                n_epochs=config["n_epochs"], gamma=config["gamma"],
                gae_lambda=0.95, clip_range=config["clip_range"],
                target_kl=config["target_kl"], ent_coef=config["entropy_start"],
                vf_coef=config["vf_coef"], max_grad_norm=0.5,
                device="cuda", verbose=1,
                tensorboard_log="./logs/", seed=config["seed"],
            )
            print("Initialized fresh policy with unchanged Pythia-70M + LoRA architecture", flush=True)
        elif resume:
            model = PPO.load(str(best_model_path), env=env, device="cuda")
        else:
            source = args.checkpoint
            if not source.exists(): raise FileNotFoundError(source)
            model = PPO.load(str(source), env=env, device="cuda")
            model.set_env(env)
            if args.logit_scale <= 0:
                raise ValueError("--logit-scale must be positive")
            if args.logit_scale != 1.0:
                # Temperature-calibrate the legacy action distribution.  This
                # leaves every deterministic decision unchanged while making
                # PPO's sampled rollout resemble the policy that was evaluated.
                with torch.no_grad():
                    model.policy.action_net.weight.mul_(args.logit_scale)
                    model.policy.action_net.bias.mul_(args.logit_scale)
                print(f"Applied action-logit scale {args.logit_scale:g}", flush=True)
            if args.side_logit_bias:
                # Equal offsets contain no directional or action-label signal;
                # they only prevent rare steering actions from being starved
                # before reward-driven PPO can separate left from right.
                with torch.no_grad():
                    model.policy.action_net.bias[1] += args.side_logit_bias
                    model.policy.action_net.bias[3] += args.side_logit_bias
                print(
                    f"Applied equal steering exploration bias "
                    f"{args.side_logit_bias:g}",
                    flush=True,
                )
        # PPO.load restores the seed embedded in the checkpoint and may seed
        # the attached vector environment as part of setup.  Apply the stage
        # seed afterwards so independent fine-tuning runs are truly distinct.
        model.seed = args.seed
        model.set_random_seed(args.seed)
        env.seed(args.seed)
        # Preserve the existing Pythia/LoRA representation and adapt only the
        # existing PPO decision/value heads.  This changes no architecture and
        # all weight updates still come exclusively from PPO rewards.
        for parameter in model.policy.parameters():
            parameter.requires_grad = False
        feature_modules = [model.policy.features_extractor.residual]
        if args.fresh:
            feature_modules.extend([
                model.policy.features_extractor.input_adapter,
                model.policy.features_extractor.output_adapter,
            ])
        trainable_modules = tuple(feature_modules) + (
            model.policy.mlp_extractor.policy_net,
            model.policy.mlp_extractor.value_net,
            model.policy.action_net,
            model.policy.value_net,
        )
        for module in trainable_modules:
            for parameter in module.parameters():
                parameter.requires_grad = True
        if args.fresh:
            for name, parameter in model.policy.features_extractor.named_parameters():
                if "lora_" in name:
                    parameter.requires_grad = True
        trainable_parameters = [
            parameter for parameter in model.policy.parameters()
            if parameter.requires_grad
        ]
        # Keep the optimizer parameter-group shape identical to a freshly
        # loaded SB3 policy.  Frozen parameters receive no gradients, but
        # including them makes every saved checkpoint reloadable.
        model.policy.optimizer = model.policy.optimizer_class(
            model.policy.parameters(),
            lr=config["learning_rate"],
            **model.policy.optimizer_kwargs,
        )
        print(
            f"Trainable policy parameters: "
            f"{sum(parameter.numel() for parameter in trainable_parameters):,}",
            flush=True,
        )
        # PPO.load restores the old optimizer settings from the baseline.
        # Fine-tuning must explicitly use the conservative Phase 2 schedule.
        model.learning_rate = config["learning_rate"]
        model.lr_schedule = lambda _: config["learning_rate"]
        model.clip_range = lambda _: config["clip_range"]
        model.target_kl = config["target_kl"]
        model.vf_coef = config["vf_coef"]
        model.n_epochs = config["n_epochs"]
        model.batch_size = config["batch_size"]
        model.ent_coef = config["entropy_start"]
        for group in model.policy.optimizer.param_groups:
            group["lr"] = config["learning_rate"]
        run_dir = Path("runs/phase2/randomized") / (
            f"p{config['physics_version']}_r{config['reward_version']}_"
            f"from_{model.num_timesteps:08d}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        eval_callback = BroadEvalCallback(
            eval_freq=max(args.eval_interval // n_envs, 1), run_dir=run_dir,
            eval_envs=config["eval_envs"], control_repeat=args.control_repeat,
        )
        baseline_reward, baseline_success = -np.inf, 0.0
        baseline_pad_departures = np.inf
        baseline_core_success = 0.0
        if not args.smoke:
            baseline_model = model
            loaded_protected_baseline = False
            if (
                (args.fresh or not resume)
                and (args.fresh or args.checkpoint.resolve() != best_model_path.resolve())
                and best_model_path.exists()
            ):
                baseline_model = PPO.load(str(best_model_path), device="cuda")
                loaded_protected_baseline = True
            baseline = evaluate_model(
                baseline_model, EVAL_SEEDS,
                num_envs=config["eval_envs"], challenge=1.0,
                control_repeat=args.control_repeat,
                canonical_initial_side="positive",
            )
            baseline_core = evaluate_model(
                baseline_model, EVAL_SEEDS,
                num_envs=config["eval_envs"], challenge=0.0,
                control_repeat=args.control_repeat,
                canonical_initial_side="positive",
            )
            if loaded_protected_baseline:
                del baseline_model
                gc.collect()
                torch.cuda.empty_cache()
            baseline_reward = baseline["mean_reward"]
            baseline_success = baseline["success_rate"]
            baseline_pad_departures = _pad_departure_failures(baseline)
            baseline_core_success = baseline_core["success_rate"]
            print(format_eval("Promotion baseline", baseline), flush=True)
            print(format_eval("Core baseline", baseline_core), flush=True)
        callback_base = model.num_timesteps if resume else 0
        if args.adaptive_curriculum:
            challenge_callback = AdaptiveChallengeCallback(
                args.challenge_start, args.challenge_end,
                target_success=args.curriculum_target,
                window_episodes=args.curriculum_window,
            )
        else:
            challenge_callback = ContinuousChallengeCallback(
                args.challenge_start, args.challenge_end, steps, callback_base
            )
        entropy_callback = EntropyScheduleCallback(
            args.entropy_start, args.entropy_end, steps, callback_base
        )
        # BroadEvalCallback already saves the exact evaluated policy every
        # 8,192 steps; redundant 2,048-step snapshots consumed tens of GB.
        callbacks = [challenge_callback, entropy_callback, eval_callback]
        model.learn(total_timesteps=steps, reset_num_timesteps=not resume, callback=callbacks, progress_bar=False)
        out = Path("rocket_pythia_phase2_smoke.zip") if args.smoke else model_path
        model.save(str(out))
        if not args.smoke and eval_callback.records:
            # Prevent stage-by-stage promotion from cumulatively eroding the
            # solved compact task.  A candidate must stay within three points
            # of its source and never fall below 95% on the held-out core set.
            core_floor = max(0.95, baseline_core_success - 0.03)
            baseline_lateral = _lateral_boundary_failures(baseline)
            eligible = [
                item for item in eval_callback.records
                if item["core_success_rate"] >= core_floor
                and _pad_departure_failures(item) <= baseline_pad_departures
                and _lateral_boundary_failures(item) <= baseline_lateral
            ]
            candidate = max(
                eligible,
                key=lambda item: (
                    item["success_rate"],
                    -_pad_departure_failures(item),
                    -_lateral_boundary_failures(item),
                    item["mean_reward"],
                ),
                default=None,
            )
            improves_broad = candidate is not None and (
                candidate["success_rate"], candidate["mean_reward"]
            ) > (baseline_success, baseline_reward)
            if improves_broad and Path(candidate["checkpoint"]).exists():
                best_success = candidate["success_rate"]
                best_reward = candidate["mean_reward"]
                checkpoint = Path(candidate["checkpoint"])
                best_pad_departures = _pad_departure_failures(candidate)
                best_lateral = _lateral_boundary_failures(candidate)
                best_model_path.write_bytes(checkpoint.read_bytes())
                # The sidecar describes the promoted checkpoint, not merely
                # the latest attempted run.  Rejected experiments must leave
                # best-model provenance untouched.
                config_path.write_text(json.dumps(config, indent=2) + "\n")
                print(
                    f"Promoted {checkpoint}: reward={best_reward:.2f} "
                    f"success={best_success:.0%} "
                    f"pad_departures={best_pad_departures} "
                    f"lateral_exits={best_lateral}"
                )
            else:
                print(
                    f"No checkpoint passed broad improvement + core retention "
                    f"(required core >= {core_floor:.1%}); kept existing best model"
                )
        print("Saved latest", out, "randomized starts", "run", run_dir)
    finally:
        env.close(); gc.collect(); torch.cuda.empty_cache()

if __name__ == "__main__": main()
