"""
Evaluate the trained policy on a single 24-hour deterministic episode and
print the same metrics the baseline benchmark prints, so the trained policy
slots directly into the comparison table.

Run from the OrbitalSched project root:

    python evaluate_policy.py

By default this loads ./checkpoints/final_policy.zip and uses seed 42 on a
24-hour (86400 second) episode, matching the baseline benchmark. Override
with environment variables:

    POLICY_PATH=./checkpoints/best_model.zip python evaluate_policy.py
    EPISODE_SECONDS=3600 python evaluate_policy.py
    SEED=99 python evaluate_policy.py
"""

from __future__ import annotations

import os
import time

import numpy as np
from stable_baselines3 import PPO

from orbitalsched.simulator.environment import OrbitalSchedulerEnv


def main() -> None:
    policy_path = os.environ.get("POLICY_PATH", "./checkpoints/final_policy.zip")
    episode_seconds = float(os.environ.get("EPISODE_SECONDS", "86400"))
    seed = int(os.environ.get("SEED", "42"))

    print(f"Loading policy from {policy_path}")
    model = PPO.load(policy_path, device="cpu")
    env = OrbitalSchedulerEnv(n_satellites=10, episode_seconds=episode_seconds)
    obs, _ = env.reset(seed=seed)
    assert env.dc is not None

    print(f"Running trained policy on {episode_seconds / 3600:.1f}h episode, seed={seed}")
    total_reward = 0.0
    n_steps = int(episode_seconds / env.dc.dt)
    t0 = time.time()
    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, _, _ = env.step(action)
        total_reward += reward
        if terminated:
            break
    wall_s = time.time() - t0

    completed = len(env.dc.completed_jobs)
    missed = len(env.dc.missed_jobs)
    pending = len(env.dc.pending_jobs)
    sla = completed / max(1, completed + missed)
    avg_chip_c = float(np.mean([s.chip_temp_c for s in env.dc.satellites]))

    print()
    header = (
        f"{'scheduler':<24}{'reward':>10}{'done':>8}{'miss':>7}"
        f"{'pend':>7}{'SLA%':>9}{'energy_Wh':>12}{'avg_chip_C':>12}"
    )
    print(header)
    print("-" * len(header))
    print(
        f"{'TrainedPPO':<24}{total_reward:>10.1f}{completed:>8d}{missed:>7d}"
        f"{pending:>7d}{sla * 100:>8.2f}%{env.dc.total_energy_wh:>12.1f}"
        f"{avg_chip_c:>11.1f}C"
    )
    print()
    print(f"Wall time: {wall_s:.1f}s")


if __name__ == "__main__":
    main()
