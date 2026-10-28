"""Stable PPO entry point for the Pythia rocket controller."""
import argparse, gc, json
from pathlib import Path
import torch
from gymnasium.wrappers import FrameStackObservation
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from pythia_controller import PythiaController
from rocket_env import RocketLandingEnv

CONFIG = {
    "physics_version": 3, "reward_version": 4, "observation_shape": [8, 9],
    "action_n": 4, "stack_size": 8, "gamma": 0.99, "seed": 42,
    "num_envs": 2, "n_steps": 256, "batch_size": 64, "n_epochs": 4,
    "learning_rate": 1e-4, "clip_range": 0.2, "target_kl": 0.03,
    "ent_coef": 0.01, "vf_coef": 0.5, "max_grad_norm": 0.5,
}
MODEL_PATH = Path("rocket_pythia_stable_latest.zip")
CONFIG_PATH = Path("rocket_pythia_stable_config.json")
RUNS_DIR = Path("runs/stable")

def make_env():
    env = FrameStackObservation(RocketLandingEnv(gamma=CONFIG["gamma"]), CONFIG["stack_size"])
    return Monitor(env, info_keywords=("is_success",))

def compatible_resume():
    try:
        return MODEL_PATH.exists() and CONFIG_PATH.exists() and json.loads(CONFIG_PATH.read_text()) == CONFIG
    except (OSError, json.JSONDecodeError):
        return False

def build_model(env, device):
    return PPO("MlpPolicy", env, policy_kwargs=dict(
        features_extractor_class=PythiaController, net_arch=dict(pi=[64], vf=[64])),
        learning_rate=CONFIG["learning_rate"], n_steps=CONFIG["n_steps"],
        batch_size=CONFIG["batch_size"], n_epochs=CONFIG["n_epochs"], gamma=CONFIG["gamma"],
        gae_lambda=0.95, clip_range=CONFIG["clip_range"], target_kl=CONFIG["target_kl"],
        ent_coef=CONFIG["ent_coef"], vf_coef=CONFIG["vf_coef"], max_grad_norm=CONFIG["max_grad_norm"],
        device=device, verbose=1, tensorboard_log="./logs/")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20_480)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", type=Path, default=None,
                        help="explicit PPO checkpoint, useful for continuing from best_model.zip")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected")
    n_envs, total_steps = (1, 128) if args.smoke else (CONFIG["num_envs"], args.steps)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"GPU: {torch.cuda.get_device_name(0)} | envs: {n_envs} | steps: {total_steps}")
    train_env = SubprocVecEnv([make_env for _ in range(n_envs)], start_method="spawn")
    eval_env = DummyVecEnv([make_env])
    train_env.seed(CONFIG["seed"]); eval_env.seed(CONFIG["seed"] + 1000)
    model = None
    explicit_resume = args.resume is not None and args.resume.exists()
    resume = explicit_resume or compatible_resume()
    try:
        if resume:
            resume_path = args.resume if explicit_resume else MODEL_PATH
            print(f"Resuming checkpoint: {resume_path}")
            model = PPO.load(str(resume_path), env=train_env, device="cuda")
        else:
            if MODEL_PATH.exists(): print("Stable checkpoint/config mismatch; starting an isolated run.")
            model = build_model(train_env, "cuda")
        run_dir = RUNS_DIR / f"from_{model.num_timesteps:08d}"; run_dir.mkdir(parents=True, exist_ok=True)
        callbacks = [CheckpointCallback(save_freq=max(2048 // n_envs, 1), save_path=str(run_dir), name_prefix="rocket"),
                     EvalCallback(eval_env, best_model_save_path=str(run_dir), log_path=str(run_dir),
                                  eval_freq=max(4096 // n_envs, 1), n_eval_episodes=10, deterministic=True, verbose=1)]
        model.learn(total_timesteps=total_steps, reset_num_timesteps=not resume, callback=callbacks, progress_bar=False)
        output_path = Path("rocket_pythia_stable_smoke.zip") if args.smoke else MODEL_PATH
        model.save(str(output_path))
        if not args.smoke:
            CONFIG_PATH.write_text(json.dumps(CONFIG, indent=2) + "\n")
        print(f"Saved {'smoke check' if args.smoke else 'latest'}: {output_path}; best evaluation is under {run_dir}")
    except KeyboardInterrupt:
        if model is not None: model.save("rocket_pythia_stable_interrupted")
        raise
    finally:
        train_env.close(); eval_env.close(); gc.collect(); torch.cuda.empty_cache()

if __name__ == "__main__": main()
