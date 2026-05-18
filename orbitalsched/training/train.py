"""
PPO training entry point for OrbitalSched.

Runs on a single H100 Lightning.ai Studio. The trained policy checkpoint
is written to ./checkpoints/policy.zip and tensorboard logs to ./tb_logs.

Usage:
    python -m orbitalsched.training.train --total-timesteps 10000000 --n-envs 16
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from orbitalsched.simulator.environment import OrbitalSchedulerEnv


def make_env(seed: int):
    """Factory for a single environment instance with a unique seed."""

    def _init():
        env = OrbitalSchedulerEnv(n_satellites=10, episode_seconds=86400.0)
        env.reset(seed=seed)
        return env

    return _init


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--log-dir", type=str, default="./tb_logs")
    parser.add_argument("--resume", type=str, default="")
    args = parser.parse_args()

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    print(
        "CUDA available:",
        torch.cuda.is_available(),
        "Device:",
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "cpu",
    )

    env = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])
    env = VecMonitor(env, filename=os.path.join(args.log_dir, "monitor.csv"))

    eval_env = SubprocVecEnv([make_env(99)])
    eval_env = VecMonitor(eval_env)

    if args.resume:
        model = PPO.load(args.resume, env=env, device="cuda")
        print(f"Resumed from {args.resume}")
    else:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            device="cuda",
            tensorboard_log=args.log_dir,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            policy_kwargs=dict(net_arch=[256, 256]),
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(200_000 // args.n_envs, 1),
        save_path=args.checkpoint_dir,
        name_prefix="ppo_orbital",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=args.checkpoint_dir,
        log_path=args.log_dir,
        eval_freq=max(100_000 // args.n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_cb, eval_cb],
    )
    final_path = os.path.join(args.checkpoint_dir, "final_policy.zip")
    model.save(final_path)
    print(f"Training complete. Final policy saved to {final_path}")


if __name__ == "__main__":
    main()
