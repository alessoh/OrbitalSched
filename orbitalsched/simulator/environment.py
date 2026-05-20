"""
Gymnasium environment for the OrbitalSched Stage One prototype, v2.

Patched from v1 to address three issues identified in the May 19, 2026
project review.

1. Compute capacity model. v1 marked jobs complete in one timestep if the
   assigned satellite's commanded load was above 1%, regardless of the
   job's compute_pflops requirement. v2 tracks per-satellite throughput
   in petaFLOP-seconds, distributes available work across each
   satellite's queued jobs in deadline order, and only completes a job
   when its accumulated work meets its compute requirement.

2. Thermal realism. v1 had every satellite radiating to deep space with
   only 50 W of solar input, equilibrating around -140 C. v2 adds
   parasitic bus heat, an Earth-IR exchange term on the Earth-facing
   side of the radiator, an albedo absorption term when sun-facing, a
   radiator sky-factor, and a simple survival heater. Steady-state chip
   temperatures now fall in the realistic LEO inference-satellite range.

3. Per-satellite priority in the action space. v1's policy only set load
   levels; the actual job-to-satellite assignment was done by a greedy
   thermal-margin heuristic inside the simulator, so the policy did not
   in fact decide which satellite ran which job. v2 doubles the action
   space, with the second half being per-satellite priority scores that
   drive assignment. The policy now genuinely controls scheduling.

The action space is backward incompatible with v1 trained checkpoints.
Retrain from scratch.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Physical constants
EARTH_RADIUS_KM = 6378.137
EARTH_MU_KM3_S2 = 398600.4418
STEFAN_BOLTZMANN = 5.670374419e-8  # W m^-2 K^-4
SOLAR_CONSTANT_W_M2 = 1361.0
EARTH_EFFECTIVE_TEMP_K = 255.0  # for IR exchange with Earth
EARTH_ALBEDO_FRACTION = 0.30
SECONDS_PER_DAY = 86400


@dataclass
class SatelliteConfig:
    """Static configuration for a single satellite."""

    sat_id: int
    semi_major_axis_km: float = EARTH_RADIUS_KM + 550.0
    inclination_deg: float = 53.0
    raan_deg: float = 0.0
    mean_anomaly_deg: float = 0.0

    # Compute and power
    compute_capacity_pflops: float = 1.0  # one H100-equivalent at full load
    payload_power_w: float = 1000.0
    parasitic_bus_power_w: float = 150.0  # avionics + comms + ACS, always on
    survival_heater_w: float = 250.0  # activates below survival_heater_setpoint_c
    survival_heater_setpoint_c: float = -10.0

    # Solar and battery
    solar_array_area_m2: float = 8.0
    solar_array_efficiency: float = 0.30
    battery_capacity_wh: float = 2000.0

    # Thermal network
    radiator_area_m2: float = 3.0
    radiator_emissivity: float = 0.85
    radiator_absorptivity_solar: float = 0.20
    radiator_sky_factor: float = 0.85  # fraction of radiator hemisphere viewing deep space
    chip_thermal_mass_jk: float = 200.0
    radiator_thermal_mass_jk: float = 4000.0
    chip_radiator_conductance_w_k: float = 12.0
    chip_max_temp_c: float = 95.0
    chip_throttle_temp_c: float = 80.0


@dataclass
class SatelliteState:
    """Dynamic state of a single satellite."""

    config: SatelliteConfig
    chip_temp_c: float = 20.0
    radiator_temp_c: float = 0.0
    battery_soc: float = 0.9
    current_load: float = 0.0  # commanded fraction of compute capacity in [0, 1]
    effective_load: float = 0.0  # achieved load after thermal throttling
    priority_score: float = 0.5  # policy's assignment priority for new jobs
    queued_jobs: int = 0
    eclipse: bool = False
    sun_facing_earth: bool = False  # earth-facing side sees albedo
    heater_active: bool = False
    position_eci_km: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class Job:
    """A single inference job awaiting scheduling."""

    job_id: int
    arrival_time_s: float
    deadline_s: float
    compute_pflops: float  # total work to complete the job, in petaFLOPs
    customer: str
    priority: int  # 1 (low) through 3 (high)
    assigned_satellite: int | None = None
    work_completed_pflops: float = 0.0
    completion_time_s: float | None = None


class WorkloadGenerator:
    """Generates a stream of synthetic inference jobs."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.next_job_id = 0
        self.arrival_rate_per_s = 0.60  # ~50k jobs per 24h episode

    def step(self, t_current: float, dt: float) -> list[Job]:
        expected = self.arrival_rate_per_s * dt
        n_arrivals = self.rng.poisson(expected)
        jobs: list[Job] = []
        for _ in range(n_arrivals):
            customer = self.rng.choice(
                ["eo", "defense", "llm"], p=[0.55, 0.10, 0.35]
            )
            if customer == "eo":
                compute_pflops = float(self.rng.uniform(0.05, 1.5))
                deadline_offset = float(self.rng.uniform(900, 3600))
                priority = 1
            elif customer == "defense":
                compute_pflops = float(self.rng.uniform(0.01, 0.2))
                deadline_offset = float(self.rng.uniform(30, 120))
                priority = 3
            else:
                compute_pflops = float(self.rng.uniform(0.005, 0.05))
                deadline_offset = float(self.rng.uniform(5, 60))
                priority = 2
            jobs.append(
                Job(
                    job_id=self.next_job_id,
                    arrival_time_s=t_current,
                    deadline_s=t_current + deadline_offset,
                    compute_pflops=compute_pflops,
                    customer=str(customer),
                    priority=int(priority),
                )
            )
            self.next_job_id += 1
        return jobs


def propagate_keplerian(
    cfg: SatelliteConfig, t_seconds: float
) -> tuple[np.ndarray, bool, bool]:
    """Return ECI position in km, eclipse flag, and sun-facing-earth flag.

    The sun-facing-earth flag is True when the satellite is in sunlight
    and on the side of Earth between the satellite and the sun, which is
    the geometry under which the Earth-facing side of the radiator
    receives meaningful albedo flux.
    """
    a = cfg.semi_major_axis_km
    n = math.sqrt(EARTH_MU_KM3_S2 / (a ** 3))
    mean_anom = math.radians(cfg.mean_anomaly_deg) + n * t_seconds
    inc = math.radians(cfg.inclination_deg)
    raan = math.radians(cfg.raan_deg)
    x_orb = a * math.cos(mean_anom)
    y_orb = a * math.sin(mean_anom)
    cos_i = math.cos(inc)
    sin_i = math.sin(inc)
    cos_o = math.cos(raan)
    sin_o = math.sin(raan)
    x = cos_o * x_orb - sin_o * cos_i * y_orb
    y = sin_o * x_orb + cos_o * cos_i * y_orb
    z = sin_i * y_orb
    pos = np.array([x, y, z])
    # Sun in -x direction (simplification: vernal equinox geometry).
    sun_dir = np.array([-1.0, 0.0, 0.0])
    along_sun = float(np.dot(pos, sun_dir))
    perp = pos - along_sun * sun_dir
    in_shadow = along_sun > 0 and float(np.linalg.norm(perp)) < EARTH_RADIUS_KM
    # Sun-facing-earth: sat is in sunlight and on sun side of earth so
    # the earth-facing surface sees the lit hemisphere.
    sun_facing_earth = (not in_shadow) and along_sun < 0
    return pos, bool(in_shadow), bool(sun_facing_earth)


class OrbitalDataCenter:
    """The simulated constellation."""

    def __init__(self, n_satellites: int = 10, seed: int = 0):
        self.n_satellites = n_satellites
        self.t_seconds = 0.0
        self.dt = 30.0
        self.workload = WorkloadGenerator(seed=seed)
        self.pending_jobs: list[Job] = []
        self.completed_jobs: list[Job] = []
        self.missed_jobs: list[Job] = []
        self.total_energy_wh = 0.0
        self.last_step_energy_wh = 0.0
        self.satellites: list[SatelliteState] = []
        for i in range(n_satellites):
            plane = i // 5
            slot = i % 5
            cfg = SatelliteConfig(
                sat_id=i,
                raan_deg=plane * 60.0,
                mean_anomaly_deg=slot * 72.0,
            )
            self.satellites.append(SatelliteState(config=cfg))

    def _assign_pending_jobs(self) -> None:
        """Assign each unassigned pending job to a satellite, ranking
        candidate satellites by the policy's per-satellite priority score
        with thermal headroom as a tiebreaker."""
        unassigned = [j for j in self.pending_jobs if j.assigned_satellite is None]
        if not unassigned:
            return
        # Sort jobs: high priority first, then earliest deadline.
        unassigned.sort(key=lambda j: (-j.priority, j.deadline_s))
        # Pre-rank candidate satellites once for this step.
        sat_indices = list(range(self.n_satellites))
        sat_indices.sort(
            key=lambda i: (
                -self.satellites[i].priority_score,
                -(
                    self.satellites[i].config.chip_max_temp_c
                    - self.satellites[i].chip_temp_c
                ),
            )
        )
        for job in unassigned:
            for i in sat_indices:
                sat = self.satellites[i]
                # Skip satellites that are unable to take any new work.
                if sat.chip_temp_c >= sat.config.chip_max_temp_c:
                    continue
                if sat.current_load < 0.01:
                    continue
                job.assigned_satellite = i
                sat.queued_jobs += 1
                break

    def _do_compute_work(self) -> int:
        """Distribute each satellite's available throughput across its
        assigned jobs in deadline order. Returns the number of jobs that
        completed during this step."""
        completed_count = 0
        sat_queues: list[list[Job]] = [[] for _ in range(self.n_satellites)]
        for job in self.pending_jobs:
            if job.assigned_satellite is not None:
                sat_queues[job.assigned_satellite].append(job)
        for q in sat_queues:
            q.sort(key=lambda j: j.deadline_s)
        for i, sat in enumerate(self.satellites):
            available_pflop_s = (
                sat.effective_load
                * sat.config.compute_capacity_pflops
                * self.dt
            )
            for job in sat_queues[i]:
                if available_pflop_s <= 1e-9:
                    break
                remaining = job.compute_pflops - job.work_completed_pflops
                if remaining <= 0.0:
                    continue
                work_this_step = min(remaining, available_pflop_s)
                job.work_completed_pflops += work_this_step
                available_pflop_s -= work_this_step
                if job.work_completed_pflops >= job.compute_pflops - 1e-9:
                    job.completion_time_s = self.t_seconds + self.dt
                    completed_count += 1
                    sat.queued_jobs = max(0, sat.queued_jobs - 1)
        return completed_count

    def step(self, scheduling_action: np.ndarray) -> dict[str, Any]:
        """Advance one simulation step.

        scheduling_action has shape (2 * n_satellites,). The first
        n_satellites entries are commanded load levels in [0, 1]. The
        next n_satellites entries are per-satellite priority scores in
        [0, 1] used to rank candidate satellites during assignment.
        """
        action = np.asarray(scheduling_action, dtype=np.float32).flatten()
        if action.shape[0] != 2 * self.n_satellites:
            raise ValueError(
                f"Expected action of shape ({2 * self.n_satellites},), "
                f"got {action.shape}."
            )
        loads = np.clip(action[: self.n_satellites], 0.0, 1.0)
        priorities = np.clip(
            action[self.n_satellites : 2 * self.n_satellites], 0.0, 1.0
        )

        # 1. Generate new arrivals.
        new_jobs = self.workload.step(self.t_seconds, self.dt)
        self.pending_jobs.extend(new_jobs)

        # 2. Apply commanded load and priority. Effective load is reduced
        #    by thermal throttling near the chip's throttle setpoint.
        for i, sat in enumerate(self.satellites):
            sat.current_load = float(loads[i])
            sat.priority_score = float(priorities[i])
            if sat.chip_temp_c > sat.config.chip_throttle_temp_c:
                throttle_factor = max(
                    0.0,
                    1.0
                    - (sat.chip_temp_c - sat.config.chip_throttle_temp_c)
                    / (
                        sat.config.chip_max_temp_c
                        - sat.config.chip_throttle_temp_c
                    ),
                )
                sat.effective_load = sat.current_load * throttle_factor
            else:
                sat.effective_load = sat.current_load

        # 3. Assign any unassigned pending jobs.
        self._assign_pending_jobs()

        # 4. Advance orbital and thermal state.
        #
        # The chip thermal time constant (mass / conductance) is roughly
        # 17 s at our default parameters, which is shorter than the 30 s
        # main timestep. Explicit Euler at the main step is therefore
        # unstable and produces nonphysical excursions (e.g. chip below
        # absolute zero). We sub-step the thermal integration with an
        # internal step of 2 s, which is well inside the stability
        # envelope for both chip and radiator nodes.
        prev_energy = self.total_energy_wh
        n_substeps = 15
        sub_dt = self.dt / n_substeps
        for sat in self.satellites:
            pos, in_eclipse, sun_facing = propagate_keplerian(
                sat.config, self.t_seconds
            )
            sat.position_eci_km = pos
            sat.eclipse = in_eclipse
            sat.sun_facing_earth = sun_facing

            # Constant-during-step quantities.
            payload_heat_w = sat.effective_load * sat.config.payload_power_w
            bus_heat_w = sat.config.parasitic_bus_power_w
            earth_facing_area = (
                1.0 - sat.config.radiator_sky_factor
            ) * sat.config.radiator_area_m2
            if sun_facing:
                albedo_w = (
                    sat.config.radiator_absorptivity_solar
                    * SOLAR_CONSTANT_W_M2
                    * EARTH_ALBEDO_FRACTION
                    * earth_facing_area
                )
            else:
                albedo_w = 0.0
            direct_solar_w = 0.0 if in_eclipse else 30.0

            heater_active_any = False
            heater_energy_wh = 0.0

            for _ in range(n_substeps):
                # Proportional survival heater: ramps from 0 W at the
                # setpoint to full power 10 C below it. Avoids the
                # bang-bang oscillation a hard threshold would cause.
                setpoint = sat.config.survival_heater_setpoint_c
                band = 10.0
                if sat.chip_temp_c < setpoint:
                    frac = min(1.0, (setpoint - sat.chip_temp_c) / band)
                    heater_w = sat.config.survival_heater_w * frac
                    heater_active_any = True
                else:
                    heater_w = 0.0

                chip_to_rad_w = sat.config.chip_radiator_conductance_w_k * (
                    sat.chip_temp_c - sat.radiator_temp_c
                )
                net_chip_heat_w = (
                    payload_heat_w + bus_heat_w + heater_w - chip_to_rad_w
                )

                radiator_t_k = sat.radiator_temp_c + 273.15
                rad_to_space_w = (
                    sat.config.radiator_emissivity
                    * STEFAN_BOLTZMANN
                    * sat.config.radiator_area_m2
                    * sat.config.radiator_sky_factor
                    * (radiator_t_k ** 4)
                )
                rad_net_to_earth_w = (
                    sat.config.radiator_emissivity
                    * STEFAN_BOLTZMANN
                    * earth_facing_area
                    * (radiator_t_k ** 4 - EARTH_EFFECTIVE_TEMP_K ** 4)
                )
                net_radiator_heat_w = (
                    chip_to_rad_w
                    + albedo_w
                    + direct_solar_w
                    - rad_to_space_w
                    - rad_net_to_earth_w
                )

                sat.chip_temp_c += (
                    net_chip_heat_w * sub_dt / sat.config.chip_thermal_mass_jk
                )
                sat.radiator_temp_c += (
                    net_radiator_heat_w
                    * sub_dt
                    / sat.config.radiator_thermal_mass_jk
                )
                heater_energy_wh += heater_w * sub_dt / 3600.0

            sat.heater_active = heater_active_any

            # Power balance over the full main step.
            solar_w = (
                0.0
                if in_eclipse
                else (
                    SOLAR_CONSTANT_W_M2
                    * sat.config.solar_array_area_m2
                    * sat.config.solar_array_efficiency
                )
            )
            payload_bus_energy_wh = (
                payload_heat_w + bus_heat_w
            ) * self.dt / 3600.0
            solar_energy_wh = solar_w * self.dt / 3600.0
            net_energy_wh = (
                solar_energy_wh - payload_bus_energy_wh - heater_energy_wh
            )
            sat.battery_soc = float(
                np.clip(
                    sat.battery_soc
                    + net_energy_wh / sat.config.battery_capacity_wh,
                    0.0,
                    1.0,
                )
            )
            self.total_energy_wh += payload_bus_energy_wh + heater_energy_wh
        self.last_step_energy_wh = self.total_energy_wh - prev_energy

        # 5. Do compute work and complete jobs.
        completed_this_step = self._do_compute_work()

        # 6. Sweep: move completed jobs out, expire missed.
        missed_this_step = 0
        remaining: list[Job] = []
        for job in self.pending_jobs:
            if job.completion_time_s is not None:
                self.completed_jobs.append(job)
                continue
            if self.t_seconds > job.deadline_s:
                self.missed_jobs.append(job)
                missed_this_step += 1
                if job.assigned_satellite is not None:
                    sat = self.satellites[job.assigned_satellite]
                    sat.queued_jobs = max(0, sat.queued_jobs - 1)
                continue
            remaining.append(job)
        self.pending_jobs = remaining

        self.t_seconds += self.dt
        return {
            "completed": completed_this_step,
            "missed": missed_this_step,
            "pending": len(self.pending_jobs),
            "energy_wh": self.total_energy_wh,
            "step_energy_wh": self.last_step_energy_wh,
        }


class OrbitalSchedulerEnv(gym.Env):
    """Gymnasium environment wrapping the orbital data center simulator."""

    metadata = {"render_modes": []}

    def __init__(
        self, n_satellites: int = 10, episode_seconds: float = 86400.0
    ):
        super().__init__()
        self.n_satellites = n_satellites
        self.episode_seconds = episode_seconds
        # Action: per-satellite load in [0, 1], then per-satellite priority in [0, 1].
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(2 * n_satellites,), dtype=np.float32
        )
        # Observation: per-satellite [chip_temp_norm, radiator_temp_norm,
        # battery_soc, effective_load, eclipse_flag, queue_norm] + global
        # [pending_norm, time_of_day].
        obs_dim = n_satellites * 6 + 2
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(obs_dim,), dtype=np.float32
        )
        self.dc: OrbitalDataCenter | None = None

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.dc = OrbitalDataCenter(
            n_satellites=self.n_satellites, seed=seed or 0
        )
        return self._observation(), {}

    def _observation(self) -> np.ndarray:
        assert self.dc is not None
        obs: list[float] = []
        for sat in self.dc.satellites:
            obs.extend(
                [
                    sat.chip_temp_c / 100.0,
                    (sat.radiator_temp_c + 50.0) / 100.0,
                    sat.battery_soc,
                    sat.effective_load,
                    1.0 if sat.eclipse else 0.0,
                    min(sat.queued_jobs / 50.0, 1.0),
                ]
            )
        obs.append(min(len(self.dc.pending_jobs) / 200.0, 1.0))
        obs.append((self.dc.t_seconds % SECONDS_PER_DAY) / SECONDS_PER_DAY)
        return np.array(obs, dtype=np.float32)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.dc is not None
        info = self.dc.step(action)
        obs = self._observation()
        completed = info["completed"]
        missed = info["missed"]
        thermal_penalty = sum(
            max(0.0, sat.chip_temp_c - sat.config.chip_throttle_temp_c) / 20.0
            for sat in self.dc.satellites
        )
        # Idle-with-backlog penalty: charges a small cost when there are
        # pending jobs but a satellite is unloaded despite having thermal
        # headroom and battery margin. This closes the v1 loophole where
        # the policy could earn reward by running at near-zero load.
        backlog = len(self.dc.pending_jobs)
        idle_penalty = 0.0
        if backlog > 5:
            for sat in self.dc.satellites:
                if (
                    sat.effective_load < 0.1
                    and sat.chip_temp_c < sat.config.chip_throttle_temp_c
                    and sat.battery_soc > 0.3
                ):
                    idle_penalty += 0.1
        reward = (
            completed * 1.0
            - missed * 5.0
            - thermal_penalty * 0.1
            - info["step_energy_wh"] * 0.001
            - idle_penalty
        )
        terminated = self.dc.t_seconds >= self.episode_seconds
        truncated = False
        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        pass


def _smoke_test() -> None:
    env = OrbitalSchedulerEnv(n_satellites=10, episode_seconds=3600.0)
    obs, _ = env.reset(seed=42)
    total_reward = 0.0
    # Action: medium load, uniform medium priority across all satellites.
    action = np.concatenate(
        [
            np.full(10, 0.5, dtype=np.float32),
            np.full(10, 0.5, dtype=np.float32),
        ]
    )
    for _ in range(120):
        obs, reward, terminated, _, info = env.step(action)
        total_reward += reward
        if terminated:
            break
    print(f"Smoke test complete. Total reward: {total_reward:.2f}")
    assert env.dc is not None
    print(f"Completed jobs: {len(env.dc.completed_jobs)}")
    print(f"Missed jobs:    {len(env.dc.missed_jobs)}")
    print(f"Pending jobs:   {len(env.dc.pending_jobs)}")
    print(f"Total energy:   {env.dc.total_energy_wh:.1f} Wh")
    print()
    print("Per-satellite end state:")
    for sat in env.dc.satellites:
        eclipse_str = "eclipse" if sat.eclipse else "sun    "
        heater_str = "heater" if sat.heater_active else "      "
        print(
            f"  SAT-{sat.config.sat_id:02d}: "
            f"chip={sat.chip_temp_c:6.1f}C, "
            f"rad={sat.radiator_temp_c:6.1f}C, "
            f"bat={sat.battery_soc * 100:5.1f}%, "
            f"load={sat.effective_load * 100:5.1f}%, "
            f"queue={sat.queued_jobs:2d}, "
            f"{eclipse_str}, {heater_str}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        _smoke_test()
    else:
        print("Use --smoke-test to verify install.", file=sys.stderr)
        sys.exit(1)
