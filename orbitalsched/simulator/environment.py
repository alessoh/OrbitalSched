"""
Gymnasium environment for the OrbitalSched Stage One prototype.

Simulates a 10-satellite low Earth orbit constellation with thermal,
power, and workload dynamics. Designed for Proximal Policy Optimization
training on a single H100.
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
SECONDS_PER_DAY = 86400


@dataclass
class SatelliteConfig:
    """Static configuration for a single satellite."""

    sat_id: int
    semi_major_axis_km: float = EARTH_RADIUS_KM + 550.0
    inclination_deg: float = 53.0
    raan_deg: float = 0.0
    mean_anomaly_deg: float = 0.0
    payload_power_w: float = 1000.0
    solar_array_area_m2: float = 8.0
    solar_array_efficiency: float = 0.30
    battery_capacity_wh: float = 2000.0
    radiator_area_m2: float = 3.0
    radiator_emissivity: float = 0.85
    chip_thermal_mass_jk: float = 200.0
    radiator_thermal_mass_jk: float = 4000.0
    chip_max_temp_c: float = 95.0
    chip_throttle_temp_c: float = 80.0


@dataclass
class SatelliteState:
    """Dynamic state of a single satellite."""

    config: SatelliteConfig
    chip_temp_c: float = 25.0
    radiator_temp_c: float = -10.0
    battery_soc: float = 0.9
    current_load: float = 0.0  # fraction of payload power in use
    queued_jobs: int = 0
    eclipse: bool = False
    position_eci_km: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class Job:
    """A single inference job awaiting scheduling."""

    job_id: int
    arrival_time_s: float
    deadline_s: float
    compute_pflops: float
    customer: str  # "eo", "defense", or "llm"
    priority: int  # 1 (low) through 3 (high)
    assigned_satellite: int | None = None
    completion_time_s: float | None = None


class WorkloadGenerator:
    """Generates a stream of synthetic inference jobs."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.next_job_id = 0
        self.arrival_rate_per_s = 0.12  # ~10k jobs per 24h episode

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
) -> tuple[np.ndarray, bool]:
    """Return ECI position in km and whether the satellite is in eclipse."""
    a = cfg.semi_major_axis_km
    n = math.sqrt(EARTH_MU_KM3_S2 / (a ** 3))  # mean motion rad/s
    mean_anom = math.radians(cfg.mean_anomaly_deg) + n * t_seconds
    # Circular-orbit approximation; eccentric anomaly = mean anomaly.
    inc = math.radians(cfg.inclination_deg)
    raan = math.radians(cfg.raan_deg)
    x_orb = a * math.cos(mean_anom)
    y_orb = a * math.sin(mean_anom)
    z_orb = 0.0
    # Rotate to ECI: inclination, then RAAN.
    cos_i = math.cos(inc)
    sin_i = math.sin(inc)
    cos_o = math.cos(raan)
    sin_o = math.sin(raan)
    x = cos_o * x_orb - sin_o * cos_i * y_orb
    y = sin_o * x_orb + cos_o * cos_i * y_orb
    z = sin_i * y_orb
    pos = np.array([x, y, z])
    # Eclipse: sun in -x ECI direction (simplification at vernal equinox).
    sun_dir = np.array([-1.0, 0.0, 0.0])
    along_sun = np.dot(pos, sun_dir)
    perp = pos - along_sun * sun_dir
    in_shadow = along_sun > 0 and np.linalg.norm(perp) < EARTH_RADIUS_KM
    return pos, bool(in_shadow)


class OrbitalDataCenter:
    """The simulated 10-satellite constellation."""

    def __init__(self, n_satellites: int = 10, seed: int = 0):
        self.n_satellites = n_satellites
        self.t_seconds = 0.0
        self.dt = 30.0  # 30-second step
        self.workload = WorkloadGenerator(seed=seed)
        self.pending_jobs: list[Job] = []
        self.completed_jobs: list[Job] = []
        self.missed_jobs: list[Job] = []
        self.total_energy_wh = 0.0
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

    def step(self, scheduling_action: np.ndarray) -> dict[str, Any]:
        # 1. Generate new jobs.
        new_jobs = self.workload.step(self.t_seconds, self.dt)
        self.pending_jobs.extend(new_jobs)

        # 2. Apply scheduling action: each satellite gets a target load level.
        for i, sat in enumerate(self.satellites):
            target_load = float(np.clip(scheduling_action[i], 0.0, 1.0))
            # Thermal throttle: reduce target if too hot.
            if sat.chip_temp_c > sat.config.chip_throttle_temp_c:
                throttle = max(
                    0.0,
                    1.0
                    - (sat.chip_temp_c - sat.config.chip_throttle_temp_c)
                    / (sat.config.chip_max_temp_c - sat.config.chip_throttle_temp_c),
                )
                target_load *= throttle
            sat.current_load = target_load

        # 3. Advance orbital and thermal state.
        for sat in self.satellites:
            pos, in_eclipse = propagate_keplerian(sat.config, self.t_seconds)
            sat.position_eci_km = pos
            sat.eclipse = in_eclipse
            # Thermal update.
            power_in_w = sat.current_load * sat.config.payload_power_w
            # Radiator emits to deep space.
            radiator_t_k = sat.radiator_temp_c + 273.15
            radiated_w = (
                sat.config.radiator_emissivity
                * STEFAN_BOLTZMANN
                * sat.config.radiator_area_m2
                * (radiator_t_k ** 4)
            )
            # Solar input on radiator (small, simplification).
            solar_on_radiator = 0.0 if in_eclipse else 50.0
            net_chip_heat = power_in_w - 0.5 * (
                sat.chip_temp_c - sat.radiator_temp_c
            )
            net_radiator_heat = (
                0.5 * (sat.chip_temp_c - sat.radiator_temp_c)
                + solar_on_radiator
                - radiated_w
            )
            sat.chip_temp_c += (
                net_chip_heat * self.dt / sat.config.chip_thermal_mass_jk
            )
            sat.radiator_temp_c += (
                net_radiator_heat * self.dt / sat.config.radiator_thermal_mass_jk
            )
            # Power update.
            if in_eclipse:
                solar_w = 0.0
            else:
                solar_w = (
                    1361.0
                    * sat.config.solar_array_area_m2
                    * sat.config.solar_array_efficiency
                )
            net_power_w = solar_w - power_in_w
            energy_delta_wh = net_power_w * self.dt / 3600.0
            sat.battery_soc = float(
                np.clip(
                    sat.battery_soc
                    + energy_delta_wh / sat.config.battery_capacity_wh,
                    0.0,
                    1.0,
                )
            )
            self.total_energy_wh += abs(power_in_w * self.dt / 3600.0)

        # 4. Try to complete pending jobs.
        completed_this_step = 0
        missed_this_step = 0
        remaining: list[Job] = []
        for job in self.pending_jobs:
            assigned_sat = job.assigned_satellite
            if assigned_sat is None:
                # Greedy assign to the satellite with most thermal headroom
                # whose current load can accommodate the job. The RL policy's
                # action already shaped the loads, so we honor that here.
                best = -1
                best_margin = -1.0
                for i, sat in enumerate(self.satellites):
                    margin = sat.config.chip_max_temp_c - sat.chip_temp_c
                    if sat.current_load > 0.01 and margin > best_margin:
                        best = i
                        best_margin = margin
                if best >= 0:
                    job.assigned_satellite = best
                    self.satellites[best].queued_jobs += 1
            # Completion logic: jobs complete when assigned and load allows.
            if job.assigned_satellite is not None:
                sat = self.satellites[job.assigned_satellite]
                if sat.current_load > 0.01:
                    job.completion_time_s = self.t_seconds + self.dt
                    self.completed_jobs.append(job)
                    sat.queued_jobs = max(0, sat.queued_jobs - 1)
                    completed_this_step += 1
                    continue
            if self.t_seconds > job.deadline_s:
                self.missed_jobs.append(job)
                missed_this_step += 1
                continue
            remaining.append(job)
        self.pending_jobs = remaining

        self.t_seconds += self.dt
        return {
            "completed": completed_this_step,
            "missed": missed_this_step,
            "pending": len(self.pending_jobs),
            "energy_wh": self.total_energy_wh,
        }


class OrbitalSchedulerEnv(gym.Env):
    """Gymnasium environment wrapping the orbital data center simulator."""

    metadata = {"render_modes": []}

    def __init__(self, n_satellites: int = 10, episode_seconds: float = 86400.0):
        super().__init__()
        self.n_satellites = n_satellites
        self.episode_seconds = episode_seconds
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(n_satellites,), dtype=np.float32
        )
        # Observation: per-satellite [chip_temp_norm, radiator_temp_norm,
        # battery_soc, current_load, eclipse_flag, queue_norm] + global
        # [pending_norm, time_of_day].
        obs_dim = n_satellites * 6 + 2
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(obs_dim,), dtype=np.float32
        )
        self.dc: OrbitalDataCenter | None = None
        self._prev_energy_wh = 0.0

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.dc = OrbitalDataCenter(
            n_satellites=self.n_satellites, seed=seed or 0
        )
        self._prev_energy_wh = 0.0
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
                    sat.current_load,
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
        # Reward shaping: reward completions, penalize missed deadlines,
        # penalize energy use, penalize thermal margin loss.
        thermal_penalty = sum(
            max(0.0, sat.chip_temp_c - sat.config.chip_throttle_temp_c) / 20.0
            for sat in self.dc.satellites
        )
        energy_delta = info["energy_wh"] - self._prev_energy_wh
        self._prev_energy_wh = info["energy_wh"]
        reward = (
            completed * 1.0
            - missed * 5.0
            - thermal_penalty * 0.1
            - energy_delta * 0.001
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
    for _ in range(120):
        action = np.full(10, 0.5, dtype=np.float32)
        obs, reward, terminated, _, info = env.step(action)
        total_reward += reward
        if terminated:
            break
    print(f"Smoke test complete. Total reward: {total_reward:.2f}")
    assert env.dc is not None
    print(f"Completed jobs: {len(env.dc.completed_jobs)}")
    print(f"Missed jobs: {len(env.dc.missed_jobs)}")
    print(f"Total energy: {env.dc.total_energy_wh:.1f} Wh")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        _smoke_test()
    else:
        print("Use --smoke-test to verify install.", file=sys.stderr)
        sys.exit(1)
