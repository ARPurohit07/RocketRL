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

from pythia_controller import PythiaController, PythiaDualAxis
from rocket3d_env import Rocket3DEnv
from rocket3d_eval import evaluate


BASELINE = Path("checkpoints/baseline.zip")
IMPROVED = Path("checkpoints/improved.zip")
METRICS = Path("runs/3d_metrics.jsonl")
EVAL_SEEDS = tuple(range(1_000_000, 1_000_064))


def make_env(challenge, focus_axis=None):
    return lambda: Monitor(FrameStackObservation(
        Rocket3DEnv(challenge=challenge, focus_axis=focus_axis), 8
    ))


def transfer_2d_representation(target, source):
    """Copy shared Pythia/heads and map the 2D state into the 3D state."""
    target_state = target.policy.state_dict()
    source_state = source.policy.state_dict()
    copied = 0
    with torch.no_grad():
        for name, current in target_state.items():
            old = source_state.get(name)
            if old is not None and old.shape == current.shape:
                target_state[name] = old.to(current.device)
                copied += 1
        for name in (
            "features_extractor.input_adapter.0.weight",
            "features_extractor.residual.0.weight",
        ):
            if name not in target_state or name not in source_state:
                continue
            current = target_state[name].clone()
            current.zero_()
            old = source_state[name].to(current.device)
            columns = {
                0: ((0, 1.0),), 1: ((2, 1.0),),
                2: ((3, 1.0),), 3: ((5, 1.0),),
                4: ((6, -1.0),), 5: ((11, -3.0 / 5.0),),
                6: ((13, 0.5), (15, 0.5)),
                7: ((14, 0.5), (16, 0.5)),
                8: ((17, 1.0),),
            }
            for old_index, destinations in columns.items():
                for new_index, scale in destinations:
                    current[:, new_index] += old[:, old_index] * scale
            target_state[name] = current
        for suffix in ("weight", "bias"):
            name = f"action_net.{suffix}"
            old = source_state[name].to(target_state[name].device)
            # MultiDiscrete([engine, steer_x, steer_y]) has 2+3+3 logits.
            target_state[name] = torch.stack((
                old[0], old[1:].mean(dim=0),
                old[1], old[2], old[3],
                old[1], old[2], old[3],
            ))
        target.policy.load_state_dict(target_state)
    print(f"Transferred {copied} equal-shaped parameter tensors from the 2D checkpoint", flush=True)


def initialize_baseline(env, source_path, seed, device):
    if BASELINE.exists():
        return
    source = PPO.load(str(source_path), device="cpu")
    model = PPO(
        "MlpPolicy", env,
        policy_kwargs={
            "features_extractor_class": PythiaController,
            "net_arch": {"pi": [64], "vf": [64]},
        },
        learning_rate=3e-5, n_steps=256, batch_size=256,
        n_epochs=4, gamma=0.99, gae_lambda=0.95,
        clip_range=0.1, ent_coef=0.01, vf_coef=0.2,
        device=device, seed=seed, verbose=0,
    )
    transfer_2d_representation(model, source)
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(BASELINE))
    print(f"Created 3D baseline: {BASELINE}", flush=True)


def initialize_dual_baseline(env, seed):
    """Factorize a transferred 3D policy into two shared Pythia axis views."""
    source = PPO.load(str(BASELINE), device="cpu")
    target = PPO(
        "MlpPolicy", env,
        policy_kwargs={
            "features_extractor_class": PythiaDualAxis,
            "net_arch": {"pi": [128], "vf": [128]},
        },
        learning_rate=5e-6, n_steps=256, batch_size=256,
        n_epochs=4, gamma=0.99, gae_lambda=0.95,
        clip_range=0.05, ent_coef=0.001, vf_coef=0.2,
        device="cpu", seed=seed, verbose=0,
    )
    old_state = source.policy.state_dict()
    state = target.policy.state_dict()
    with torch.no_grad():
        for name, current in state.items():
            old = old_state.get(name)
            if old is not None and old.shape == current.shape:
                state[name] = old.clone()
        for name in (
            "features_extractor.input_adapter.0.weight",
            "features_extractor.residual.0.weight",
        ):
            old = old_state[name]
            mapped = torch.zeros_like(state[name])
            mapped[:, 0] = old[:, 0]
            mapped[:, 1] = old[:, 2]
            mapped[:, 2] = old[:, 3]
            mapped[:, 3] = old[:, 5]
            mapped[:, 4] = -old[:, 6]
            mapped[:, 5] = -old[:, 11] * (5.0 / 3.0)
            mapped[:, 6] = old[:, 13] + old[:, 15]
            mapped[:, 7] = old[:, 14] + old[:, 16]
            mapped[:, 8] = old[:, 17]
            state[name] = mapped
        for path in ("policy_net", "value_net"):
            weight_name = f"mlp_extractor.{path}.0.weight"
            bias_name = f"mlp_extractor.{path}.0.bias"
            old_weight = old_state[weight_name]
            new_weight = torch.zeros_like(state[weight_name])
            new_weight[:64, :128] = old_weight
            new_weight[64:, 128:] = old_weight
            state[weight_name] = new_weight
            state[bias_name] = old_state[bias_name].repeat(2)
        old_action = old_state["action_net.weight"]
        action = torch.zeros_like(state["action_net.weight"])
        action[0, :64] = old_action[0] * 0.5
        action[0, 64:] = old_action[0] * 0.5
        action[1, :64] = old_action[1] * 0.5
        action[1, 64:] = old_action[1] * 0.5
        action[2:5, :64] = old_action[2:5]
        action[5:8, 64:] = old_action[2:5]
        state["action_net.weight"] = action
        old_bias = old_state["action_net.bias"]
        state["action_net.bias"] = torch.cat((old_bias[:5], old_bias[2:5]))
        state["value_net.weight"] = old_state["value_net.weight"].repeat(1, 2) * 0.5
        target.policy.load_state_dict(state)
    full = evaluate(target, EVAL_SEEDS, challenge=1.0)
    near = evaluate(target, EVAL_SEEDS, challenge=0.25)
    core = evaluate(target, EVAL_SEEDS[:32], challenge=0.0)
    print(f"Dual-axis baseline: full={full} near={near} core={core}", flush=True)
    target.save(str(BASELINE))
    print(f"Saved dual-axis baseline to {BASELINE}", flush=True)


class TrainingMonitor(BaseCallback):
    def __init__(
        self, total_steps, start_challenge, eval_interval, core_floor=0.75,
        focus_axis=None,
    ):
        super().__init__(verbose=0)
        self.total_steps = int(total_steps)
        self.start_challenge = float(start_challenge)
        self.eval_interval = int(eval_interval)
        self.best_score = (-1.0, -1.0, -1.0, -np.inf)
        self.core_floor = core_floor
        self.focus_axis = focus_axis
        self.last_bucket = -1
        self.challenge = start_challenge
        self.outcomes = []

    def _on_training_start(self):
        full = evaluate(self.model, EVAL_SEEDS, challenge=1.0)
        near = evaluate(self.model, EVAL_SEEDS, challenge=0.25)
        core = evaluate(self.model, EVAL_SEEDS[:32], challenge=0.0)
        focus = self._focus_eval(self.model)
        self.best_score = self._score(full, near, core, focus)
        print(f"Starting 3D checkpoint: full={full} near={near} core={core}", flush=True)
        if focus is not None:
            print(f"Starting focused benchmark: {focus}", flush=True)
        if IMPROVED.exists():
            previous = PPO.load(str(IMPROVED), device="cpu")
            prior_full = evaluate(previous, EVAL_SEEDS, challenge=1.0)
            prior_near = evaluate(previous, EVAL_SEEDS, challenge=0.25)
            prior_core = evaluate(previous, EVAL_SEEDS[:32], challenge=0.0)
            prior_focus = self._focus_eval(previous)
            self.best_score = max(
                self.best_score,
                self._score(prior_full, prior_near, prior_core, prior_focus),
            )
            print(f"Protected improved checkpoint: full={prior_full['success_rate']:.1%} near={prior_near['success_rate']:.1%} core={prior_core['success_rate']:.1%}", flush=True)
            del previous

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
            if done and worker < 4:
                self.outcomes.append(int(info.get("is_success", False)))
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
            print(f"Training step={self.num_timesteps} recent_success={rate:.1%} challenge={self.challenge:.3f}", flush=True)
        if self.num_timesteps % self.eval_interval:
            return True
        full = evaluate(self.model, EVAL_SEEDS, challenge=1.0)
        near = evaluate(self.model, EVAL_SEEDS, challenge=0.25)
        core = evaluate(self.model, EVAL_SEEDS[:32], challenge=0.0)
        focus = self._focus_eval(self.model)
        record = {
            "environment_version": 3,
            "step": self.num_timesteps, "challenge": self.challenge,
            "full": full, "near": near, "core": core, "focus": focus,
        }
        METRICS.parent.mkdir(parents=True, exist_ok=True)
        with METRICS.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        focus_text = "" if focus is None else f" focus={focus['success_rate']:.1%}"
        print(f"3D held-out step={self.num_timesteps}: full={full} near={near['success_rate']:.1%} core={core['success_rate']:.1%}{focus_text}", flush=True)
        score = self._score(full, near, core, focus)
        if core["success_rate"] >= self.core_floor and score > self.best_score:
            self.model.save(str(IMPROVED))
            self.best_score = score
            print(f"Updated {IMPROVED} at {self.num_timesteps} steps", flush=True)
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initialize-from", default="rocket_pythia_phase2_best.zip")
    parser.add_argument("--init-only", action="store_true")
    parser.add_argument("--dual-init", action="store_true")
    parser.add_argument("--steps", type=int, default=262_144)
    parser.add_argument("--eval-interval", type=int, default=16_384)
    parser.add_argument("--challenge-start", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--restart-from-baseline", action="store_true")
    parser.add_argument("--logit-scale", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--decouple-axis-heads", action="store_true")
    parser.add_argument("--y-neutral-bias", type=float, default=3.0)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--focus-axis", choices=("y",))
    parser.add_argument("--y-exploration-bias", type=float)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.steps <= 0 or args.eval_interval <= 0:
        raise ValueError("Steps and evaluation interval must be positive")
    torch.set_num_threads(2)
    env = DummyVecEnv([
        make_env(challenge, focus_axis=args.focus_axis)
        for challenge in ([args.challenge_start] * 4 + [0.5] * 2 + [1.0] * 2)
    ])
    try:
        if args.dual_init:
            if not BASELINE.exists():
                raise FileNotFoundError(BASELINE)
            initialize_dual_baseline(env, args.seed)
            return
        initialize_baseline(env, args.initialize_from, args.seed, args.device)
        if args.init_only:
            return
        source = BASELINE if args.restart_from_baseline else (
            IMPROVED if IMPROVED.exists() else BASELINE
        )
        model = PPO.load(str(source), env=env, device=args.device)
        model.set_random_seed(args.seed)
        if args.logit_scale != 1.0:
            with torch.no_grad():
                model.policy.action_net.weight.mul_(args.logit_scale)
                model.policy.action_net.bias.mul_(args.logit_scale)
        if args.decouple_axis_heads:
            # The 2D transfer copies X steering logits to Y, causing both
            # axes to make almost identical decisions.  Initialize the Y
            # channel independently so PPO can discover its own control law.
            with torch.no_grad():
                model.policy.action_net.weight[5:8].normal_(0.0, 0.05)
                model.policy.action_net.bias[5:8] = torch.tensor(
                    [0.0, args.y_neutral_bias, 0.0], device=model.policy.device
                )
                first = model.policy.features_extractor.input_adapter[0].weight
                residual = model.policy.features_extractor.residual[0].weight
                for column in (1, 4, 7, 10):
                    first[:, column].normal_(0.0, 0.02)
                    residual[:, column].normal_(0.0, 0.05)
            print("Initialized independent Y-axis steering head", flush=True)
        elif args.y_exploration_bias is not None:
            with torch.no_grad():
                model.policy.action_net.bias[6] = args.y_exploration_bias
            print(
                f"Set Y neutral logit to {args.y_exploration_bias:g}",
                flush=True,
            )
        if args.probe_only:
            probe_challenge = args.challenge_start if args.focus_axis else 0.0
            deterministic = evaluate(
                model, EVAL_SEEDS, challenge=probe_challenge,
                focus_axis=args.focus_axis,
            )
            stochastic = evaluate(
                model, EVAL_SEEDS, challenge=probe_challenge,
                focus_axis=args.focus_axis, deterministic=False,
            )
            print(
                f"Y-neutral bias={args.y_exploration_bias if args.y_exploration_bias is not None else args.y_neutral_bias} "
                f"probe deterministic={deterministic['success_rate']:.1%} "
                f"stochastic={stochastic['success_rate']:.1%}",
                flush=True,
            )
            return
        env.seed(args.seed)
        for parameter in model.policy.parameters():
            parameter.requires_grad = False
        modules = (
            model.policy.features_extractor.input_adapter,
            model.policy.features_extractor.output_adapter,
            model.policy.features_extractor.residual,
            model.policy.mlp_extractor,
            model.policy.action_net,
            model.policy.value_net,
        )
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad = True
        model.policy.optimizer = model.policy.optimizer_class(
            model.policy.parameters(), lr=args.learning_rate,
            **model.policy.optimizer_kwargs,
        )
        model.learning_rate = args.learning_rate
        model.lr_schedule = lambda _: args.learning_rate
        model.clip_range = lambda _: 0.05
        model.ent_coef = 0.005
        model.n_epochs = 4
        model.batch_size = 256
        callback = TrainingMonitor(
            args.steps, args.challenge_start, args.eval_interval,
            focus_axis=args.focus_axis,
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
