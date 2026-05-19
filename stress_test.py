"""
Stress test for OrbitalSched baselines.

The default workload (~0.054 pFLOPS demand against 10 pFLOPS supply at full
load) is over-provisioned by ~180x, so all three baselines hit 100% SLA on
the easy benchmark. To find a regime where the schedulers actually
differentiate, we sweep two knobs:

1. Per-satellite compute capacity, lowered to tighten the supply side.
2. Workload arrival rate, raised to thicken the demand side.

A learned policy has room to beat the heuristics only in regimes where
the heuristics themselves start to struggle. This script identifies those
regimes empirically so the next training run is aimed at a problem worth
solving.

Run from the OrbitalSched project root:

    python stress_test.py
"""

from __future__ import annotations

import time

from orbitalsched.simulator.environment import OrbitalSchedulerEnv
from orbitalsched.scheduler.baselines import (
    EDFThrottleScheduler,
    MILPScheduler,
    PriorityEDFScheduler,
)

SCHEDULER_CLASSES = [
    EDFThrottleScheduler,
    PriorityEDFScheduler,
    MILPScheduler,
]


def run_one(
    scheduler_cls,
    capacity_pflops: float = 1.0,
    arrival_rate_per_s: float = 0.12,
    episode_seconds: float = 3600.0,
    seed: int = 42,
) -> dict:
    """Run a single (scheduler, capacity, rate) combination and return metrics.

    Mutates each satellite's config after construction so the dataclass
    default does not get in the way. Mutates the workload generator's
    arrival rate the same way.
    """
    scheduler = scheduler_cls(n_satellites=10)
    env = OrbitalSchedulerEnv(n_satellites=10, episode_seconds=episode_seconds)
    env.reset(seed=seed)
    assert env.dc is not None

    for sat in env.dc.satellites:
        sat.config.compute_capacity_pflops = capacity_pflops
    env.dc.workload.arrival_rate_per_s = arrival_rate_per_s

    scheduler.reset()
    total_reward = 0.0
    n_steps = int(episode_seconds / env.dc.dt)
    t0 = time.time()
    for _ in range(n_steps):
        action = scheduler.predict(env.dc)
        _, reward, terminated, _, _ = env.step(action)
        total_reward += reward
        if terminated:
            break
    wall_s = time.time() - t0

    completed = len(env.dc.completed_jobs)
    missed = len(env.dc.missed_jobs)
    pending = len(env.dc.pending_jobs)
    sla = completed / max(1, completed + missed)
    return {
        "scheduler": scheduler_cls.__name__,
        "capacity": capacity_pflops,
        "rate": arrival_rate_per_s,
        "reward": total_reward,
        "completed": completed,
        "missed": missed,
        "pending": pending,
        "sla": sla,
        "energy_wh": env.dc.total_energy_wh,
        "wall_s": wall_s,
    }


def sweep(label: str, settings: list[dict]) -> None:
    """Run every scheduler at each setting in `settings` and print a table."""
    print(f"=== {label} ===")
    header = (
        f"{'config':<26}{'scheduler':<22}"
        f"{'reward':>10}{'done':>7}{'miss':>6}{'pend':>6}"
        f"{'SLA%':>8}{'energy_Wh':>11}{'wall_s':>8}"
    )
    print(header)
    print("-" * len(header))
    for s in settings:
        config_label = (
            f"cap={s.get('capacity_pflops', 1.0):.3f} "
            f"rate={s.get('arrival_rate_per_s', 0.12):.2f}"
        )
        for cls in SCHEDULER_CLASSES:
            r = run_one(
                cls,
                capacity_pflops=s.get("capacity_pflops", 1.0),
                arrival_rate_per_s=s.get("arrival_rate_per_s", 0.12),
            )
            print(
                f"{config_label:<26}{r['scheduler']:<22}"
                f"{r['reward']:>10.1f}{r['completed']:>7d}{r['missed']:>6d}"
                f"{r['pending']:>6d}{r['sla'] * 100:>7.2f}%"
                f"{r['energy_wh']:>11.1f}{r['wall_s']:>7.1f}s"
            )
        print()


def main() -> None:
    print(
        "OrbitalSched baseline stress test, 1-hour episodes, seed=42.\n"
        "Baseline ordering inside each block: EDF, PriorityEDF, MILP.\n"
    )

    sweep(
        "Sweep A: reduce per-satellite compute capacity (tighten supply)",
        [
            {"capacity_pflops": 1.00},  # default, control
            {"capacity_pflops": 0.20},
            {"capacity_pflops": 0.10},
            {"capacity_pflops": 0.05},
        ],
    )

    sweep(
        "Sweep B: raise workload arrival rate (thicken demand)",
        [
            {"arrival_rate_per_s": 0.12},  # default, control
            {"arrival_rate_per_s": 0.60},
            {"arrival_rate_per_s": 1.20},
            {"arrival_rate_per_s": 2.40},
        ],
    )

    sweep(
        "Sweep C: both knobs together (the regime a learned policy could win)",
        [
            {"capacity_pflops": 0.20, "arrival_rate_per_s": 0.60},
            {"capacity_pflops": 0.20, "arrival_rate_per_s": 1.20},
            {"capacity_pflops": 0.10, "arrival_rate_per_s": 0.60},
            {"capacity_pflops": 0.10, "arrival_rate_per_s": 1.20},
        ],
    )


if __name__ == "__main__":
    main()
