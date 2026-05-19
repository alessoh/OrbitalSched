"""
Baseline schedulers for OrbitalSched benchmarking.

Three reference policies share the same action interface as the PPO
policy, so they can be benchmarked head-to-head on identical workload
traces. The action vector each baseline produces has shape
(2 * n_satellites,), matching the v2 OrbitalSchedulerEnv: the first
half is per-satellite commanded load in [0, 1], the second half is
per-satellite assignment priority in [0, 1].

The three baselines:

    EDFThrottleScheduler   Earliest-deadline-first with thermal
                           throttling. The straightforward heuristic
                           any real operator would compare against.

    PriorityEDFScheduler   EDF with customer-priority weighting and
                           a reserved cold satellite for urgent jobs.

    MILPScheduler          Rolling-horizon mixed integer linear
                           program. Solves a small assignment LP
                           every K simulator steps. Approximates an
                           oracle upper bound, given perfect knowledge
                           of the current pending queue.

Run from the command line to benchmark all three on a one-hour episode:

    python -m orbitalsched.scheduler.baselines

For a 24-hour episode pass --episode-seconds 86400. To skip the MILP
(useful if PuLP or CBC is unavailable on the host), pass --skip-milp.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

try:
    import pulp
    _HAVE_PULP = True
except ImportError:
    _HAVE_PULP = False

from orbitalsched.simulator.environment import (
    OrbitalDataCenter,
    OrbitalSchedulerEnv,
)


class BaselineScheduler:
    """Common interface for baseline schedulers."""

    def __init__(self, n_satellites: int = 10):
        self.n_satellites = n_satellites

    def predict(self, dc: OrbitalDataCenter) -> np.ndarray:
        raise NotImplementedError

    def reset(self) -> None:
        """Called once before each episode."""
        pass


class EDFThrottleScheduler(BaselineScheduler):
    """Earliest-deadline-first with thermal throttling.

    Each satellite's commanded load scales with backlog pressure. When
    a chip exceeds its throttle setpoint, load is cut back hard so the
    chip can recover. Priority scores favor satellites with the most
    thermal headroom, so newly arriving jobs are routed there first.
    """

    def predict(self, dc: OrbitalDataCenter) -> np.ndarray:
        loads = np.zeros(self.n_satellites, dtype=np.float32)
        priorities = np.zeros(self.n_satellites, dtype=np.float32)
        backlog = len(dc.pending_jobs)
        backlog_pressure = min(1.0, backlog / 20.0)
        for i, sat in enumerate(dc.satellites):
            headroom = sat.config.chip_max_temp_c - sat.chip_temp_c
            margin_norm = max(0.0, min(1.0, headroom / 80.0))
            if sat.chip_temp_c >= sat.config.chip_throttle_temp_c:
                loads[i] = 0.1
            elif sat.queued_jobs > 0 or backlog > 0:
                loads[i] = float(min(1.0, 0.4 + 0.6 * backlog_pressure))
            else:
                loads[i] = 0.15
            priorities[i] = margin_norm
        return np.concatenate([loads, priorities])


class PriorityEDFScheduler(BaselineScheduler):
    """Priority-weighted EDF.

    Like EDF, but reserves the satellite with the most thermal headroom
    as a hot spare for high-priority jobs by giving it a priority-score
    boost when urgent work is pending.
    """

    def predict(self, dc: OrbitalDataCenter) -> np.ndarray:
        loads = np.zeros(self.n_satellites, dtype=np.float32)
        priorities = np.zeros(self.n_satellites, dtype=np.float32)
        backlog = len(dc.pending_jobs)
        backlog_pressure = min(1.0, backlog / 20.0)
        high_pri_pending = sum(1 for j in dc.pending_jobs if j.priority >= 3)

        ranked = sorted(
            range(self.n_satellites),
            key=lambda i: -(
                dc.satellites[i].config.chip_max_temp_c
                - dc.satellites[i].chip_temp_c
            ),
        )

        for rank, i in enumerate(ranked):
            sat = dc.satellites[i]
            headroom = sat.config.chip_max_temp_c - sat.chip_temp_c
            margin_norm = max(0.0, min(1.0, headroom / 80.0))
            if sat.chip_temp_c >= sat.config.chip_throttle_temp_c:
                loads[i] = 0.1
            elif sat.queued_jobs > 0 or backlog > 0:
                loads[i] = float(min(1.0, 0.4 + 0.6 * backlog_pressure))
            else:
                # Keep the coldest satellite warm so it can absorb the
                # next urgent arrival without spin-up delay.
                loads[i] = 0.25 if rank == 0 else 0.15
            boost = 0.25 if (rank == 0 and high_pri_pending > 0) else 0.0
            priorities[i] = float(min(1.0, margin_norm + boost))
        return np.concatenate([loads, priorities])


class MILPScheduler(BaselineScheduler):
    """Rolling-horizon MILP.

    Every `resolve_every_steps` simulator steps, solve an assignment
    program over the next `horizon_seconds` of look-ahead. Between
    solves, replay the cached action.

    Decision variables:
        x[j, s] in {0, 1}  : assign pending job j to satellite s
        load[s] in [0, 1]  : commanded load for satellite s

    Constraints:
        Each job assigned to at most one satellite.
        Per-satellite compute capacity over the horizon.
        Thermal: any satellite at or beyond throttle setpoint gets a
            load cap proportional to its remaining margin.

    Objective:
        Maximize sum over assigned jobs of (priority * (1 + urgency))
        minus a small load cost, so the solver prefers to assign urgent
        high-priority work and not to spin up satellites unnecessarily.
    """

    def __init__(
        self,
        n_satellites: int = 10,
        horizon_seconds: float = 60.0,
        resolve_every_steps: int = 2,
        solver_time_limit_s: float = 2.0,
    ):
        super().__init__(n_satellites)
        if not _HAVE_PULP:
            raise ImportError(
                "MILPScheduler requires `pulp`. Install with `pip install pulp`."
            )
        self.horizon_seconds = horizon_seconds
        self.resolve_every_steps = resolve_every_steps
        self.solver_time_limit_s = solver_time_limit_s
        self._cached_action: np.ndarray | None = None
        self._steps_since_solve = 0

    def reset(self) -> None:
        self._cached_action = None
        self._steps_since_solve = 0

    def predict(self, dc: OrbitalDataCenter) -> np.ndarray:
        if (
            self._cached_action is not None
            and self._steps_since_solve < self.resolve_every_steps
        ):
            self._steps_since_solve += 1
            return self._cached_action
        action = self._solve(dc)
        self._cached_action = action
        self._steps_since_solve = 1
        return action

    def _idle_action(self) -> np.ndarray:
        """Action when there is nothing urgent to schedule: low load,
        uniform priority, one warm satellite to absorb arrivals."""
        loads = np.full(self.n_satellites, 0.15, dtype=np.float32)
        loads[0] = 0.25
        priorities = np.full(self.n_satellites, 0.5, dtype=np.float32)
        return np.concatenate([loads, priorities])

    def _solve(self, dc: OrbitalDataCenter) -> np.ndarray:
        # Candidates: any pending job whose deadline falls inside an
        # extended look-ahead window. The extension lets the solver
        # consider work that does not need to start this minute but
        # will be relevant soon.
        deadline_window = dc.t_seconds + self.horizon_seconds * 3
        candidates = [
            j for j in dc.pending_jobs if j.deadline_s <= deadline_window
        ]
        if not candidates:
            return self._idle_action()

        prob = pulp.LpProblem("orbital_milp", pulp.LpMaximize)

        x = {
            (j_idx, s_idx): pulp.LpVariable(
                f"x_{j_idx}_{s_idx}", cat=pulp.LpBinary
            )
            for j_idx in range(len(candidates))
            for s_idx in range(self.n_satellites)
        }
        load = {
            s_idx: pulp.LpVariable(
                f"load_{s_idx}", lowBound=0.0, upBound=1.0
            )
            for s_idx in range(self.n_satellites)
        }

        # Each job to at most one satellite.
        for j_idx in range(len(candidates)):
            prob += (
                pulp.lpSum(
                    x[(j_idx, s_idx)] for s_idx in range(self.n_satellites)
                )
                <= 1
            )

        # Per-satellite capacity over horizon. Note the load variable
        # determines how much of the satellite's nominal capacity is
        # actually usable for new work in this horizon.
        for s_idx in range(self.n_satellites):
            sat = dc.satellites[s_idx]
            nominal_capacity = (
                sat.config.compute_capacity_pflops * self.horizon_seconds
            )
            # Already-queued work consumes capacity too. Estimate it.
            already_queued_work = sum(
                max(0.0, j.compute_pflops - j.work_completed_pflops)
                for j in dc.pending_jobs
                if j.assigned_satellite == s_idx
            )
            usable_for_new = max(
                0.0, nominal_capacity - already_queued_work
            )
            if usable_for_new <= 0.0:
                # No room: forbid new assignments to this satellite.
                for j_idx in range(len(candidates)):
                    prob += x[(j_idx, s_idx)] == 0
                continue
            prob += (
                pulp.lpSum(
                    candidates[j_idx].compute_pflops * x[(j_idx, s_idx)]
                    for j_idx in range(len(candidates))
                )
                <= load[s_idx] * usable_for_new
            )

        # Thermal caps.
        for s_idx in range(self.n_satellites):
            sat = dc.satellites[s_idx]
            if sat.chip_temp_c >= sat.config.chip_max_temp_c:
                prob += load[s_idx] == 0
            elif sat.chip_temp_c >= sat.config.chip_throttle_temp_c:
                throttle = max(
                    0.0,
                    1.0
                    - (sat.chip_temp_c - sat.config.chip_throttle_temp_c)
                    / (
                        sat.config.chip_max_temp_c
                        - sat.config.chip_throttle_temp_c
                    ),
                )
                prob += load[s_idx] <= throttle

        # Objective: urgency-weighted assignment value minus a small
        # load cost.
        value_terms = []
        for j_idx, job in enumerate(candidates):
            time_to_deadline = max(1.0, job.deadline_s - dc.t_seconds)
            urgency = 1.0 / time_to_deadline
            value = job.priority * (1.0 + urgency)
            for s_idx in range(self.n_satellites):
                value_terms.append(value * x[(j_idx, s_idx)])
        load_cost = 0.05 * pulp.lpSum(
            load[s_idx] for s_idx in range(self.n_satellites)
        )
        prob += pulp.lpSum(value_terms) - load_cost

        solver = pulp.PULP_CBC_CMD(
            msg=False, timeLimit=self.solver_time_limit_s
        )
        prob.solve(solver)

        loads_out = np.zeros(self.n_satellites, dtype=np.float32)
        for s_idx in range(self.n_satellites):
            v = load[s_idx].value()
            loads_out[s_idx] = float(v) if v is not None else 0.15

        # Priorities derived from solver decisions: satellites that the
        # MILP assigned new work to get higher priority for any further
        # arrivals that come in before the next solve.
        priorities_out = np.zeros(self.n_satellites, dtype=np.float32)
        for s_idx in range(self.n_satellites):
            assigned_count = 0
            for j_idx in range(len(candidates)):
                v = x[(j_idx, s_idx)].value()
                if v is not None and v > 0.5:
                    assigned_count += 1
            sat = dc.satellites[s_idx]
            headroom_norm = max(
                0.0,
                min(
                    1.0,
                    (sat.config.chip_max_temp_c - sat.chip_temp_c) / 80.0,
                ),
            )
            priorities_out[s_idx] = float(
                min(1.0, 0.5 * headroom_norm + 0.5 * min(1.0, assigned_count / 5.0))
            )

        return np.concatenate([loads_out, priorities_out])


def run_benchmark(
    scheduler: BaselineScheduler,
    episode_seconds: float = 86400.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Run one episode and return summary metrics."""
    env = OrbitalSchedulerEnv(
        n_satellites=scheduler.n_satellites, episode_seconds=episode_seconds
    )
    env.reset(seed=seed)
    scheduler.reset()
    assert env.dc is not None
    total_reward = 0.0
    n_steps = int(episode_seconds / env.dc.dt)
    for _ in range(n_steps):
        action = scheduler.predict(env.dc)
        _, reward, terminated, _, _ = env.step(action)
        total_reward += reward
        if terminated:
            break
    completed = len(env.dc.completed_jobs)
    missed = len(env.dc.missed_jobs)
    pending = len(env.dc.pending_jobs)
    sla = completed / max(1, completed + missed)
    avg_chip_c = float(
        np.mean([s.chip_temp_c for s in env.dc.satellites])
    )
    return {
        "scheduler": type(scheduler).__name__,
        "total_reward": total_reward,
        "completed": completed,
        "missed": missed,
        "pending": pending,
        "sla_compliance": sla,
        "energy_wh": env.dc.total_energy_wh,
        "avg_chip_c": avg_chip_c,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark baseline schedulers on OrbitalSched."
    )
    parser.add_argument("--episode-seconds", type=float, default=3600.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-milp",
        action="store_true",
        help="Skip the MILP baseline (use if PuLP or CBC is unavailable).",
    )
    args = parser.parse_args()

    schedulers: list[BaselineScheduler] = [
        EDFThrottleScheduler(n_satellites=10),
        PriorityEDFScheduler(n_satellites=10),
    ]
    if not args.skip_milp and _HAVE_PULP:
        schedulers.append(MILPScheduler(n_satellites=10))

    print(
        f"Running baselines on a {args.episode_seconds / 3600:.1f}-hour episode, "
        f"seed={args.seed}\n"
    )
    header = (
        f"{'scheduler':<24} {'reward':>10} {'done':>7} {'miss':>6} "
        f"{'pend':>6} {'SLA%':>7} {'energy_Wh':>11} {'avg_chip_C':>11}"
    )
    print(header)
    print("-" * len(header))
    for sch in schedulers:
        r = run_benchmark(
            sch, episode_seconds=args.episode_seconds, seed=args.seed
        )
        print(
            f"{r['scheduler']:<24} "
            f"{r['total_reward']:>10.1f} "
            f"{r['completed']:>7d} "
            f"{r['missed']:>6d} "
            f"{r['pending']:>6d} "
            f"{r['sla_compliance'] * 100:>6.2f}% "
            f"{r['energy_wh']:>11.1f} "
            f"{r['avg_chip_c']:>10.1f}C"
        )


if __name__ == "__main__":
    main()
