"""Pure PPO training with exactly one 3D baseline and one improved checkpoint."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from gymnasium.wrappers import FrameStackObservation
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from rocket3d_env import Rocket3DEnv
from rocket3d_eval import evaluate


BASELINE = Path("checkpoints/baseline.zip")
IMPROVED = Path("checkpoints/improved.zip")
METRICS = Path("runs/3d_metrics.jsonl")
EVAL_SEEDS = tuple(range(1_000_000, 1_000_064))


def make_env(
    challenge, focus_axis=None, height_floor=None, height_ceiling=None,
    min_start_radius=0.0,
):
    return lambda: Monitor(FrameStackObservation(
        Rocket3DEnv(
            challenge=challenge, focus_axis=focus_axis,
            height_floor=height_floor, height_ceiling=height_ceiling,
            min_start_radius=min_start_radius,
        ), 8
    ))


class TrainingMonitor(BaseCallback):
    def __init__(
        self, total_steps, start_challenge, eval_interval, core_floor=0.90,
        focus_axis=None, source=None, compare_existing=False,
        low_radius_start=1.0, low_radius_target=2.5,
        mid_radius_start=2.0, mid_radius_target=3.5,
    ):
        super().__init__(verbose=0)
        self.total_steps = int(total_steps)
        self.start_challenge = float(start_challenge)
        self.eval_interval = int(eval_interval)
        self.best_score = (-1.0, -1.0, -1.0, -np.inf)
        self.core_floor = core_floor
        self.focus_axis = focus_axis
        self.source = Path(source) if source is not None else None
        self.compare_existing = compare_existing or self.source != IMPROVED
        self.last_bucket = -1
        self.challenge = start_challenge
        self.outcomes = []
        self.low_outcomes = []
        self.mid_outcomes = []
        self.low_radius_start = low_radius_start
        self.low_radius_target = low_radius_target
        self.mid_radius_start = mid_radius_start
        self.mid_radius_target = mid_radius_target
        self.low_radius = low_radius_start
        self.mid_radius = mid_radius_start

    def _on_training_start(self):
        full = evaluate(self.model, EVAL_SEEDS, challenge=1.0)
        near = evaluate(self.model, EVAL_SEEDS, challenge=0.25)
        core = evaluate(self.model, EVAL_SEEDS[:32], challenge=0.0)
        focus = self._focus_eval(self.model)
        starting_score = self._score(full, near, core, focus)
        prior_score = starting_score if IMPROVED.exists() and not self.compare_existing else None
        print(
            f"Starting 3D checkpoint: full={full['success_rate']:.1%} "
            f"near={near['success_rate']:.1%} "
            f"core={core['success_rate']:.1%}", flush=True,
        )
        if focus is not None:
            print(f"Starting focused benchmark: {focus}", flush=True)
        if IMPROVED.exists() and self.compare_existing:
            previous = PPO.load(str(IMPROVED), device="cpu")
            prior_full = evaluate(previous, EVAL_SEEDS, challenge=1.0)
            prior_near = evaluate(previous, EVAL_SEEDS, challenge=0.25)
            prior_core = evaluate(previous, EVAL_SEEDS[:32], challenge=0.0)
            prior_focus = self._focus_eval(previous)
            prior_score = self._score(prior_full, prior_near, prior_core, prior_focus)
            print(f"Protected improved checkpoint: full={prior_full['success_rate']:.1%} near={prior_near['success_rate']:.1%} core={prior_core['success_rate']:.1%}", flush=True)
            del previous
        self.best_score = max(starting_score, prior_score) if prior_score is not None else starting_score
        if core["success_rate"] >= self.core_floor and (
            prior_score is None or starting_score > prior_score
        ):
            self.model.save(str(IMPROVED))
            print(f"Saved improved starting policy to {IMPROVED}", flush=True)

    def _focus_eval(self, model):
        if self.focus_axis is None:
            return None
        return evaluate(
            model, EVAL_SEEDS, challenge=0.6, focus_axis=self.focus_axis
        )

    def _score(self, full, near, core, focus=None):
        if focus is not None:
            return (
                focus["success_rate"], full["success_rate"],
                near["success_rate"], core["success_rate"],
                full["mean_reward"],
            )
        return (
            full["success_rate"], near["success_rate"],
            core["success_rate"], full["mean_reward"],
        )

    def _on_step(self):
        for worker, (done, info) in enumerate(zip(
            self.locals.get("dones", ()), self.locals.get("infos", ())
        )):
            if done:
                outcome = int(info.get("is_success", False))
                if worker < 4:
                    self.outcomes.append(outcome)
                elif worker < 6:
                    self.low_outcomes.append(outcome)
                else:
                    self.mid_outcomes.append(outcome)
        if len(self.outcomes) >= 64:
            rate = float(np.mean(self.outcomes))
            target = 0.40 if self.focus_axis else 0.58
            floor = 0.20 if self.focus_axis else 0.40
            if rate >= target:
                self.challenge = min(1.0, self.challenge + 0.05)
            elif rate < floor:
                self.challenge = max(
                    self.start_challenge, self.challenge - 0.0125
                )
            self.challenge = max(
                self.challenge,
                min(1.0, self.start_challenge + self.num_timesteps / self.total_steps),
            )
            self.outcomes.clear()
            for wrapped in self.training_env.envs[:4]:
                wrapped.unwrapped.set_challenge(self.challenge)
            low_rate = float(np.mean(self.low_outcomes)) if self.low_outcomes else 0.0
            mid_rate = float(np.mean(self.mid_outcomes)) if self.mid_outcomes else 0.0
            if len(self.low_outcomes) >= 16 and low_rate >= 0.55:
                self.low_radius += 0.25
            if len(self.mid_outcomes) >= 16 and mid_rate >= 0.55:
                self.mid_radius += 0.25
            progress = min(1.0, self.num_timesteps / self.total_steps)
            self.low_radius = min(self.low_radius_target, max(
                self.low_radius,
                self.low_radius_start
                + progress * (self.low_radius_target - self.low_radius_start),
            ))
            self.mid_radius = min(self.mid_radius_target, max(
                self.mid_radius,
                self.mid_radius_start
                + progress * (self.mid_radius_target - self.mid_radius_start),
            ))
            for wrapped in self.training_env.envs[4:6]:
                wrapped.unwrapped.set_min_start_radius(self.low_radius)
            for wrapped in self.training_env.envs[6:8]:
                wrapped.unwrapped.set_min_start_radius(self.mid_radius)
            print(
                f"Training step={self.num_timesteps} recent_success={rate:.1%} "
                f"low_offset_success={low_rate:.1%}@{self.low_radius:.2f}m "
                f"mid_offset_success={mid_rate:.1%}@{self.mid_radius:.2f}m "
                f"challenge={self.challenge:.3f}", flush=True,
            )
            self.low_outcomes.clear()
            self.mid_outcomes.clear()
        if self.num_timesteps % self.eval_interval:
            return True
        full = evaluate(self.model, EVAL_SEEDS, challenge=1.0)
        near = evaluate(self.model, EVAL_SEEDS, challenge=0.25)
        core = evaluate(self.model, EVAL_SEEDS[:32], challenge=0.0)
        focus = self._focus_eval(self.model)
        record = {
            "environment_version": 3,
            "step": self.num_timesteps, "challenge": self.challenge,
            "low_radius": self.low_radius, "mid_radius": self.mid_radius,
            "full": full, "near": near, "core": core, "focus": focus,
        }
        METRICS.parent.mkdir(parents=True, exist_ok=True)
        with METRICS.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        focus_text = "" if focus is None else f" focus={focus['success_rate']:.1%}"
        print(
            f"3D held-out step={self.num_timesteps}: "
            f"full={full['success_rate']:.1%} "
            f"near={near['success_rate']:.1%} "
            f"core={core['success_rate']:.1%} "
            f"failures={full['failures']}{focus_text}", flush=True,
        )
        score = self._score(full, near, core, focus)
        if core["success_rate"] >= self.core_floor and score > self.best_score:
            self.model.save(str(IMPROVED))
            self.best_score = score
            print(f"Updated {IMPROVED} at {self.num_timesteps} steps", flush=True)
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=262_144)
    parser.add_argument("--eval-interval", type=int, default=16_384)
    parser.add_argument("--challenge-start", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--restart-from-baseline", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.98)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--action-logit-scale", type=float, default=1.0)
    parser.add_argument("--position-scale", type=float, default=1.0)
    parser.add_argument("--train-lora", action="store_true")
    parser.add_argument("--train-head-only", action="store_true")
    parser.add_argument("--focus-axis", choices=("y",))
    parser.add_argument("--low-height-ceiling", type=float, default=10.0)
    parser.add_argument("--low-start-radius", type=float, default=1.0)
    parser.add_argument("--low-target-radius", type=float, default=2.5)
    parser.add_argument("--mid-height-floor", type=float, default=10.0)
    parser.add_argument("--mid-height-ceiling", type=float, default=16.0)
    parser.add_argument("--mid-start-radius", type=float, default=2.0)
    parser.add_argument("--mid-target-radius", type=float, default=3.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.steps <= 0 or args.eval_interval <= 0 or args.learning_rate <= 0:
        raise ValueError("Steps, evaluation interval and learning rate must be positive")
    if args.entropy_coef < 0 or args.action_logit_scale <= 0 or args.position_scale <= 0:
        raise ValueError("Entropy coefficient must be nonnegative and scales positive")
    if not 0 < args.gamma < 1 or not 0 < args.gae_lambda <= 1:
        raise ValueError("Gamma must be in (0, 1); GAE lambda must be in (0, 1]")
    if args.train_lora and args.train_head_only:
        raise ValueError("Choose LoRA training or head-only training")
    if not 0 <= args.low_start_radius <= args.low_target_radius:
        raise ValueError("Low-offset radius range must increase from a nonnegative start")
    if not 0 <= args.mid_start_radius <= args.mid_target_radius:
        raise ValueError("Middle-offset radius range must increase from a nonnegative start")
    torch.set_num_threads(2)
    env = DummyVecEnv([
        make_env(
            challenge, focus_axis=args.focus_axis,
            height_floor=floor, height_ceiling=ceiling,
            min_start_radius=radius,
        )
        for challenge, floor, ceiling, radius in (
            [(args.challenge_start, None, None, 0.0)] * 4
            + [(1.0, None, args.low_height_ceiling, args.low_start_radius)] * 2
            + [(
                1.0, args.mid_height_floor, args.mid_height_ceiling,
                args.mid_start_radius,
            )] * 2
        )
    ])
    try:
        if not BASELINE.exists():
            raise FileNotFoundError(f"Missing tracked baseline checkpoint: {BASELINE}")
        source = BASELINE if args.restart_from_baseline else (
            IMPROVED if IMPROVED.exists() else BASELINE
        )
        model = PPO.load(str(source), env=env, device=args.device)
        model.set_random_seed(args.seed)
        if args.action_logit_scale != 1.0:
            with torch.no_grad():
                model.policy.action_net.weight.mul_(args.action_logit_scale)
                model.policy.action_net.bias.mul_(args.action_logit_scale)
        if args.position_scale != 1.0:
            with torch.no_grad():
                extractor = model.policy.features_extractor
                extractor.input_adapter[0].weight[:, 0].mul_(args.position_scale)
                extractor.residual[0].weight[:, 0].mul_(args.position_scale)
        env.seed(args.seed)
        for parameter in model.policy.parameters():
            parameter.requires_grad = False
        modules = (
            model.policy.mlp_extractor,
            model.policy.action_net,
            model.policy.value_net,
        )
        if not args.train_head_only:
            modules += (
                model.policy.features_extractor.input_adapter,
                model.policy.features_extractor.output_adapter,
                model.policy.features_extractor.residual,
            )
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad = True
        if args.train_lora:
            lora_count = 0
            for name, parameter in model.policy.features_extractor.backbone.named_parameters():
                if "lora_" in name:
                    parameter.requires_grad = True
                    lora_count += parameter.numel()
            if lora_count == 0:
                raise RuntimeError("No Pythia LoRA parameters found to train")
            print(f"Training {lora_count:,} Pythia LoRA parameters with PPO", flush=True)
        print(
            f"Trainable policy parameters: "
            f"{sum(p.numel() for p in model.policy.parameters() if p.requires_grad):,}",
            flush=True,
        )
        model.policy.optimizer = model.policy.optimizer_class(
            model.policy.parameters(), lr=args.learning_rate,
            **model.policy.optimizer_kwargs,
        )
        model.learning_rate = args.learning_rate
        model.lr_schedule = lambda _: args.learning_rate
        model.gamma = args.gamma
        model.gae_lambda = args.gae_lambda
        model.clip_range = lambda _: 0.05
        model.ent_coef = args.entropy_coef
        model.n_epochs = 4
        model.batch_size = 256
        callback = TrainingMonitor(
            args.steps, args.challenge_start, args.eval_interval,
            focus_axis=args.focus_axis, source=source,
            compare_existing=args.position_scale != 1.0,
            low_radius_start=args.low_start_radius,
            low_radius_target=args.low_target_radius,
            mid_radius_start=args.mid_start_radius,
            mid_radius_target=args.mid_target_radius,
        )
        model.learn(
            total_timesteps=args.steps,
            reset_num_timesteps=True,
            callback=callback,
            progress_bar=False,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
