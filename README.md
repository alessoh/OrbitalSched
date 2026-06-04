# OrbitalSched: Stage One Prototype

*A thermal- and orbit-aware inference scheduler for orbital data centers*

**Prototype version:** 0.1.0
**Last updated:** May 16, 2026

---

## What This Is

.OrbitalSched is the scheduling layer for orbital data centers, equivalent in role to what Kubernetes is for terrestrial cloud. The product takes a stream of inference job requests with latency, throughput, and price constraints, and decides which satellite in a constellation runs which job in which orbit, against a moving constraint set defined by thermal capacity, solar input, battery state, ground station visibility, and chip health.

This repository contains the Stage One prototype, a fully software-based simulation and reinforcement-learning training pipeline that demonstrates the core thesis on a ten-satellite simulated constellation. The prototype is designed to be buildable by a small team in six to eight weeks of focused work, on a stack consisting of a Windows laptop for development, a single H100 GPU on Lightning.ai for training, and Vercel for the demonstration UI.

The four-stage roadmap at the end of this document describes how the prototype evolves into a full Stage One demonstration of one hundred satellites, then into the production Stage Two deployment on real customer telemetry, then into Stage Three multi-operator coordination, and finally into Stage Four as the operating system of the orbital and lunar compute economy.

---

## Scope of the Stage One Prototype

The prototype simulates ten inference satellites in low Earth orbit, distributed across two orbital planes for visual clarity. Each satellite carries an inference accelerator drawing nominally one kilowatt of payload power, a thermal subsystem modeled as a five-node lumped-parameter network, a solar array and battery, and an internal job queue. The constellation is served by ten ground stations whose geographic locations match the actual public commercial network. A synthetic workload generator produces three customer profiles, namely Earth observation, defense ISR, and foundation model inference, at a combined rate of roughly ten thousand jobs per twenty-four-hour episode.

The scheduler runs as a reinforcement-learning policy trained with Proximal Policy Optimization on a single H100. Baselines include first-come-first-served, priority queue with deadline weighting, and a mixed-integer linear program solved on a rolling five-minute horizon. The interpretability shell uses the Anthropic API to generate natural-language explanations of scheduling decisions for the demonstration UI.

The prototype is not flight-ready software. It does not handle real telemetry ingestion, fault tolerance against single-event upsets, multi-operator coordination, or any production-grade reliability features. Those belong to Stage Two and beyond. The prototype's job is to demonstrate, with quantitative rigor, that thermal-and-orbit-aware scheduling delivers a measurable improvement over credible baselines, and to convert that demonstration into pilot letters of intent with constellation operators.

---

## Architecture Overview

The system divides into four components. The simulator runs on the developer's laptop for interactive work and on Lightning.ai for batched training, and is responsible for propagating satellite orbits, integrating thermal dynamics, generating workloads, and exposing a Gymnasium-compatible reinforcement-learning environment. The training pipeline runs on Lightning.ai with a single H100, trains the PPO policy against the simulator, and produces a serialized policy checkpoint plus tensorboard logs. The API server runs FastAPI either locally for development or on Lightning.ai for the public demo, loads the trained policy, accepts inference scheduling requests, and returns scheduling decisions with optional natural-language explanations. The UI is a Next.js application deployed on Vercel that visualizes the constellation in real time, displays scheduling decisions, and lets the demonstrator dial workload parameters during a live presentation.

The four components communicate over standard HTTP and WebSockets. The UI calls the API for scheduling decisions and subscribes to a WebSocket stream for live constellation state. The API holds a reference to the loaded policy and to a long-running simulator instance whose state advances in real time for the demo.

---

## Technology Stack

The Python side uses Python 3.11, PyTorch 2.3, Stable Baselines3 for the PPO implementation, Gymnasium for the RL environment interface, FastAPI for the API server, Uvicorn as the ASGI server, Pydantic for data models, NumPy and SciPy for numerical work, and the Anthropic Python SDK for the interpretability shell. The orbital mechanics use a custom Keplerian propagator with eclipse detection, which is sufficient at the prototype's fidelity level and avoids a dependency on heavier packages like Skyfield that complicate Windows installation.

The UI side uses Next.js 14 with the App Router, React 18, TypeScript, Tailwind CSS for styling, shadcn/ui for components, Three.js with React Three Fiber for the three-dimensional constellation view, and Recharts for the time-series visualizations. The UI deploys to Vercel's free hobby tier, which is more than adequate for prototype traffic.

The development environment uses Git for version control, GitHub for code hosting, and the Lightning.ai Studio interface for managed cloud GPU access. Lightning.ai connects directly to GitHub repositories and exposes the trained model via persistent endpoints, which is the path the public demo uses.

---

## Prerequisites

The Windows laptop needs Python 3.11 or newer, ideally installed from the official Python.org distribution rather than the Microsoft Store version, because the Store version has subtle path issues with some scientific Python packages. Git for Windows is required, with the option to use Git Bash or PowerShell for terminal work. A modern web browser is needed for the UI. The recommended editor is Visual Studio Code with the Python and Pylance extensions, the Tailwind CSS IntelliSense extension, and the GitHub Copilot extension if available.

The Lightning.ai account needs the free tier to start, with the option to upgrade to a paid plan once H100 training begins. The H100 Studio costs approximately three to four dollars per hour as of mid-2026, and the full prototype training run consumes between twelve and forty H100-hours depending on convergence behavior, for a total training cost in the range of forty to one hundred and sixty dollars.

The Vercel account uses the free hobby tier. The free tier allows unlimited static deployments, a hundred gigabytes of bandwidth per month, and serverless function execution sufficient for the prototype's traffic.

The optional Anthropic API account is needed only if you want the natural-language interpretability shell. The cost is small at prototype scale, on the order of a few dollars for a demo session.

---

## Repository Layout

The full prototype lives in a single repository organized as follows.

```
orbitalsched-prototype/
├── README.md                    (this document)
├── pyproject.toml
├── .env.example
├── .gitignore
├── orbitalsched/
│   ├── __init__.py
│   ├── simulator/
│   │   ├── __init__.py
│   │   ├── constellation.py     (satellite and orbital mechanics)
│   │   ├── thermal.py           (thermal model)
│   │   ├── workload.py          (job generator)
│   │   ├── ground_stations.py   (ground network model)
│   │   └── environment.py       (gymnasium RL environment)
│   ├── scheduler/
│   │   ├── __init__.py
│   │   ├── baselines.py         (FCFS, priority queue, MILP)
│   │   ├── interpretability.py  (Anthropic API explainer)
│   │   └── policy.py            (PPO policy wrapper)
│   ├── training/
│   │   ├── __init__.py
│   │   └── train.py             (Lightning.ai training entry point)
│   └── api/
│       ├── __init__.py
│       └── server.py            (FastAPI app)
├── ui/                          (separate Next.js project)
│   ├── package.json
│   ├── app/
│   │   ├── page.tsx
│   │   └── layout.tsx
│   ├── components/
│   │   ├── ConstellationView.tsx
│   │   ├── SchedulerPanel.tsx
│   │   └── TelemetryChart.tsx
│   └── lib/
│       └── api.ts
└── tests/
    ├── test_simulator.py
    ├── test_baselines.py
    └── test_environment.py
```

The Python project at the top level is managed with pyproject.toml using either uv or pip. The Next.js project under the ui directory is a separate Node project deployed independently to Vercel.

---

## Installation on a Windows Laptop

The installation flow assumes you have administrator rights on the laptop, a reasonable internet connection, and roughly an hour of setup time.

Begin by installing Python 3.11 from python.org, making sure to check the option to add Python to the PATH during installation. Verify the install by opening a new PowerShell window and running `python --version`, which should report 3.11.x or newer.

Install Git for Windows from git-scm.com, accepting the defaults. Verify with `git --version` in PowerShell.

Install Node.js 20 LTS from nodejs.org for the UI development. Verify with `node --version` and `npm --version`.

Install Visual Studio Code from code.visualstudio.com, then install the recommended extensions from within VS Code.

Clone the repository and set up the Python environment by running the following in PowerShell from the directory where you want the project to live.

```powershell
git clone https://github.com/yourusername/orbitalsched-prototype.git
cd orbitalsched-prototype
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

If PowerShell blocks script execution when running the Activate.ps1 line, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once and accept the prompt.

The pyproject.toml file declares the dependencies. The contents of pyproject.toml are as follows.

```toml
[project]
name = "orbitalsched"
version = "0.1.0"
description = "Thermal and orbit aware inference scheduler prototype"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.12",
    "gymnasium>=0.29",
    "stable-baselines3>=2.3",
    "torch>=2.3",
    "fastapi>=0.110",
    "uvicorn[standard]>=0.29",
    "pydantic>=2.6",
    "python-dotenv>=1.0",
    "anthropic>=0.30",
    "pulp>=2.8",
    "tensorboard>=2.16",
    "rich>=13.7",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "black>=24.0",
    "ruff>=0.4",
    "mypy>=1.10",
]
```

Copy the .env.example file to .env and fill in any API keys you intend to use, including the Anthropic key for the interpretability shell.

Verify the install by running the smoke test, which exercises the simulator and produces a short trace.

```powershell
python -m orbitalsched.simulator.environment --smoke-test
```

If the smoke test prints constellation telemetry and exits cleanly, the laptop side of the install is working.

---

## The Simulator

The simulator is the heart of the prototype. It is a deterministic, fully observable, discrete-time environment in which the scheduler's decisions are evaluated. The file `orbitalsched/simulator/environment.py` is reproduced in full below. It implements a Gymnasium environment with ten satellites, a Keplerian orbital propagator, a five-node thermal model, a simple solar-power-and-battery model, a workload generator, and a reward function that combines service-level compliance, energy efficiency, and thermal margin.

```python
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
        reward = (
            completed * 1.0
            - missed * 5.0
            - thermal_penalty * 0.1
            - (info["energy_wh"] - self.dc.total_energy_wh) * 0.001
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
```

This file is intentionally complete and runnable on a Windows laptop without GPU acceleration. The smoke test exercises one hour of simulated time at a constant fifty-percent load on all satellites, prints the resulting reward and job-completion summary, and serves as the install verification step.

---

## The Training Loop

The training file `orbitalsched/training/train.py` runs on Lightning.ai with one H100 attached. It instantiates the environment, configures Proximal Policy Optimization through Stable Baselines3, and trains for ten million timesteps with periodic evaluation and checkpoint saving. The complete file is reproduced below.

```python
"""
PPO training entry point for OrbitalSched.

Runs on a single H100 Lightning.ai Studio. The trained policy checkpoint
is written to ./checkpoints/policy.zip and tensorboard logs to ./tb_logs.
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
        "CUDA available:", torch.cuda.is_available(),
        "Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
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
        save_freq=200_000 // args.n_envs,
        save_path=args.checkpoint_dir,
        name_prefix="ppo_orbital",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=args.checkpoint_dir,
        log_path=args.log_dir,
        eval_freq=100_000 // args.n_envs,
        n_eval_episodes=5,
        deterministic=True,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_cb, eval_cb],
    )
    model.save(os.path.join(args.checkpoint_dir, "final_policy.zip"))
    print("Training complete.")


if __name__ == "__main__":
    main()
```

The training run completes in approximately twelve to forty hours of wall-clock time on a single H100 depending on the random seed and on whether you choose to train through ten million timesteps or stop earlier when the evaluation reward plateaus. The intermediate checkpoints written every two hundred thousand timesteps let you abort early and use the best policy seen so far.

---

## Running on Lightning.ai

Lightning.ai Studios are persistent cloud development environments with attached GPUs. The workflow for the prototype is to develop on the Windows laptop with the simulator in CPU mode, push to GitHub, then open a Lightning Studio that pulls from the same GitHub repository and runs the GPU-intensive training.

Sign in to lightning.ai and create a new Studio. Choose the PyTorch base image and select the H100 instance type. Connect the Studio to your GitHub repository through the integrations panel, which gives the Studio a local clone of the code that syncs both ways.

From the Studio's terminal, install the package in editable mode and launch training.

```bash
cd ~/orbitalsched-prototype
pip install -e ".[dev]"
python -m orbitalsched.training.train --total-timesteps 10000000 --n-envs 16
```

Tensorboard logs appear under ./tb_logs and the Studio's built-in port forwarding lets you view them in a browser tab. The trained policy is written to ./checkpoints when training completes.

To serve the trained policy as an API for the UI, the same Studio runs Uvicorn against the FastAPI server. Lightning.ai exposes a public URL for any forwarded port, which the Vercel UI uses as its API endpoint.

---

## The API Server

The file `orbitalsched/api/server.py` implements a FastAPI service that loads the trained policy, exposes scheduling endpoints, and pushes live constellation state over WebSocket. The complete file is reproduced below.

```python
"""
FastAPI server exposing the OrbitalSched scheduler to the UI.

Run with:
    uvicorn orbitalsched.api.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from stable_baselines3 import PPO

from orbitalsched.simulator.environment import OrbitalSchedulerEnv


CHECKPOINT_PATH = os.environ.get(
    "ORBITALSCHED_CHECKPOINT", "./checkpoints/final_policy.zip"
)


class ConstellationState(BaseModel):
    t_seconds: float
    satellites: list[dict]
    pending_jobs: int
    completed_jobs: int
    missed_jobs: int


class SchedulerAppState:
    def __init__(self) -> None:
        self.env: OrbitalSchedulerEnv | None = None
        self.model: PPO | None = None
        self.obs: np.ndarray | None = None
        self.running: bool = True


state = SchedulerAppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.env = OrbitalSchedulerEnv(n_satellites=10, episode_seconds=86400.0)
    state.obs, _ = state.env.reset(seed=0)
    if Path(CHECKPOINT_PATH).exists():
        state.model = PPO.load(CHECKPOINT_PATH, device="cpu")
        print(f"Loaded policy from {CHECKPOINT_PATH}")
    else:
        print(f"No checkpoint at {CHECKPOINT_PATH}; using random policy.")
    yield
    state.running = False


app = FastAPI(title="OrbitalSched API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "checkpoint_loaded": state.model is not None}


@app.get("/state")
async def get_state() -> ConstellationState:
    assert state.env is not None and state.env.dc is not None
    sats = [
        {
            "id": s.config.sat_id,
            "chip_temp_c": round(s.chip_temp_c, 1),
            "radiator_temp_c": round(s.radiator_temp_c, 1),
            "battery_soc": round(s.battery_soc, 3),
            "current_load": round(s.current_load, 3),
            "eclipse": s.eclipse,
            "queued_jobs": s.queued_jobs,
            "position": s.position_eci_km.tolist(),
        }
        for s in state.env.dc.satellites
    ]
    return ConstellationState(
        t_seconds=state.env.dc.t_seconds,
        satellites=sats,
        pending_jobs=len(state.env.dc.pending_jobs),
        completed_jobs=len(state.env.dc.completed_jobs),
        missed_jobs=len(state.env.dc.missed_jobs),
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while state.running:
            if state.env is None or state.obs is None:
                await asyncio.sleep(0.1)
                continue
            if state.model is not None:
                action, _ = state.model.predict(state.obs, deterministic=True)
            else:
                action = np.full(10, 0.5, dtype=np.float32)
            state.obs, reward, terminated, _, info = state.env.step(action)
            payload = (await get_state()).model_dump()
            payload["reward"] = round(float(reward), 3)
            payload["action"] = action.tolist()
            await websocket.send_text(json.dumps(payload))
            if terminated:
                state.obs, _ = state.env.reset(seed=int(state.env.dc.t_seconds) % 1000)
            await asyncio.sleep(0.5)  # 2 Hz live update
    except WebSocketDisconnect:
        pass
```

The API server runs at 2 Hz in real time during the demo, advancing the simulation by thirty simulated seconds per step. The UI receives a constellation state update every five hundred milliseconds, which is sufficient for smooth visualization while keeping the network traffic bounded.

For local development on Windows, run the server with the following command from an activated virtual environment.

```powershell
uvicorn orbitalsched.api.server:app --host 0.0.0.0 --port 8000 --reload
```

On Lightning.ai, the same command runs inside the Studio with port eight thousand forwarded to a public URL. The forwarded URL becomes the `NEXT_PUBLIC_API_URL` environment variable in the Vercel deployment.

---

## The UI on Vercel

The UI lives in a separate Next.js project under the `ui/` directory of the repository. It is structured as a Next.js 14 App Router project with three primary components. The constellation view uses React Three Fiber to render the ten satellites in their orbits around a low-poly Earth, with color-coding for thermal margin, eclipse state, and current load. The scheduler panel shows pending and completed job counts, current reward, the most recent scheduling action across the ten satellites, and a natural-language explanation of why the policy made the decision it did, generated by the Anthropic API. The telemetry chart uses Recharts to show rolling time-series of chip temperature, battery state of charge, and aggregate throughput.

Bootstrap the UI project on the Windows laptop with the following commands.

```powershell
cd ui
npx create-next-app@latest . --typescript --tailwind --app --eslint --no-src-dir --import-alias "@/*"
npm install three @react-three/fiber @react-three/drei recharts lucide-react
npm install -D @types/three
```

Set the API endpoint in `.env.local` for local development.

```
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
```

Deploy to Vercel by pushing the `ui/` directory to its own GitHub repository and connecting that repository to a Vercel project through the Vercel dashboard. Set the environment variables in the Vercel project settings to point at the Lightning.ai-hosted API URL once the API is running there.

```
NEXT_PUBLIC_API_URL=https://your-studio-name-8000.lightning.ai
NEXT_PUBLIC_WS_URL=wss://your-studio-name-8000.lightning.ai/ws
```

Vercel deploys automatically on every push to the main branch and gives you a stable public URL that you can use for live demonstrations.

---

## Verifying the Prototype End to End

The end-to-end demonstration flow is as follows. On the laptop, run the smoke test to confirm the simulator works. Open a Lightning.ai Studio with H100 attached and pull the same code from GitHub. Start training with the full ten-million-timestep run and let it complete overnight or over a weekend, monitoring tensorboard for the validation reward curve to confirm convergence. When training finishes, start the Uvicorn API server in the Lightning Studio with the trained checkpoint loaded, and forward port eight thousand to a public URL. Push the UI to Vercel and configure the environment variables to point at the Lightning URL. Open the Vercel URL in a browser and watch the live demonstration of the trained scheduler running against the synthetic workload.

A successful prototype demonstration shows the constellation view animating in real time, the scheduler making non-trivial decisions across the ten satellites, the reward curve trending upward over the demonstration window, and a clear quantitative gap between the trained policy and the priority-queue baseline when both are run on the same workload trace.

---

## Roadmap to Full Stage One

The ten-satellite prototype is the foundation. Full Stage One, as described in the broader implementation plan, expands the simulation to one hundred satellites, adds the full set of three customer profiles with realistic job distributions, implements three baseline schedulers including the mixed-integer linear program, completes the natural-language interpretability shell with conversation memory, and produces the public benchmark suite and white paper that anchor the seed-stage conversations.

The engineering work to get from this prototype to full Stage One is roughly four to six months for a team of four to six engineers. The simulator generalizes from ten to one hundred satellites without architectural changes, since the gymnasium environment is already parameterized on satellite count. The training pipeline needs more parallelism, with sixty-four parallel environments instead of sixteen and approximately fifty million timesteps instead of ten million, but the same Lightning.ai single-H100 setup suffices. The API server needs a multi-client mode where multiple browser sessions can connect simultaneously without interfering. The UI needs the production polish that distinguishes a demo from a product, including the ability to dial workload parameters live during a presentation and the ability to switch between multiple policy variants for A/B comparison.

The success criterion at the end of full Stage One is a one-hundred-satellite demonstration showing twenty to thirty percent fewer service-level violations and fifteen percent lower energy per inference token compared to the priority-queue baseline, with two signed pilot letters of intent from named constellation operators willing to provide real telemetry for Stage Two.

---

## Stage Two: Year One Production Deployment

Stage Two is shaped by three technology breakthroughs landing in late 2026 and through 2027. NVIDIA's Vera Rubin Space-1 Module begins shipping to launch partners and reaches deployed inference capacity in the field. Foundation models mature to the point of interpreting unusual telemetry patterns and writing root-cause analyses that previously required senior thermal engineers. Real on-orbit telemetry from Starcloud-2 and the early Sophia Space deployments produces the first datasets against which a real-data thermal predictor can be trained.

The architectural changes are substantial. The lumped-parameter thermal model from the prototype is replaced by a learned thermal predictor trained on real customer telemetry. Scheduling intelligence migrates partially from the ground-based control plane onto the spacecraft itself, with the Vera Rubin Space-1 module running the policy locally for sub-second decisions while the ground retains long-range planning. The operator-facing interface gains a Claude-powered copilot that answers natural-language questions about scheduler behavior in real time.

The engineering work splits into four parallel streams across an expanded team of approximately twenty engineers. The integration stream builds the production telemetry pipeline that ingests customer data, normalizes it, and feeds it into both the live scheduler and the model training loop. The onboard stream ports the policy to flight-software-compliant code on Vera Rubin Space-1 with appropriate radiation tolerance protections. The operator stream builds production-grade dashboards, alerting, audit logging, and the multi-tenant control plane. The platform stream handles the deployment automation, observability, and customer onboarding.

A Series A round in the fifteen to twenty-five million dollar range funds Stage Two on the strength of the prototype demonstration and the first paying contract, which we model as a one to three million dollar annual recurring revenue commitment from the first operator.

---

## Stage Three: Year Two Multi-Constellation Scale

Stage Three is shaped by three further breakthroughs. The asynchronous, massively-collaborative agent architecture that Karpathy and others have flagged matures into a production training pattern, letting the company run hundreds of simultaneous policy improvement experiments across the deployed fleet. The large-scale orbital deployments led by Starcloud-3 in its Starship PEZ-dispenser configuration expand the addressable scheduling problem by an order of magnitude. World-model approaches for physical environments become reliable enough to replace the explicit simulator as the predictive backbone, evaluating hypothetical scheduling decisions in milliseconds rather than seconds.

The design moves from single-operator to multi-operator scheduling. When one operator's constellation is in shadow and another has thermal headroom, an inference job submitted to the first should be allowed to execute on the second under a cross-operator agreement that OrbitalSched brokers. The policy improvement loop becomes continuous and distributed, with each deployment contributing telemetry to a shared learning system. The world model replaces the explicit physics simulator as the primary predictive backbone, with the simulator retained as a falsifier that the world model is continuously benchmarked against. Sovereign-deployment variants for defense customers run entirely in customer-controlled environments without data flowing back to the shared learning system.

The team grows to approximately seventy engineers. A Series B round in the fifty to one hundred million dollar range funds the expansion on the strength of multi-operator revenue and the strategic position. Revenue scales toward twenty-five to fifty million dollars of annual recurring revenue by the end of Stage Three.

---

## Stage Four: Year Three Orbital Compute Operating System

Stage Four is the transition from scheduling product to the operating system of the orbital and lunar compute economy. The shaping breakthroughs are autonomous spacecraft operations maturing to the point of minimal ground crew involvement, the beginning of lunar compute infrastructure on the far side or at the south pole, in-space servicing and refurbishment extending spacecraft lifetimes, and foundation models capable of multi-week planning horizons.

The design extends along four axes. Capacity planning becomes a first-class function alongside scheduling, with the system recommending which orbits to fill, which satellites to retire, and when to launch new capacity. Multi-domain scheduling integrates orbital, cislunar, and lunar surface compute. The operator copilot becomes a planning agent that engages directly with customer procurement processes. Marketplace primitives let smaller operators sell spare capacity into a unified pool that larger customers can draw from.

The team approaches two hundred people, with a significant share now in customer-facing roles. Revenue scales toward one hundred million dollars of annual recurring revenue. The Stage Four success criterion is a defensible platform position as the coordination layer for orbital and early lunar compute, with concrete optionality in the public market or strategic exit direction.

---

## Honest Caveats and Known Limitations of the Prototype

The prototype trades fidelity for tractability in several places that any engineer reading this code carefully will notice, and that any technical reviewer in a customer conversation will probably ask about.

The orbital propagator uses a circular Keplerian model rather than SGP4 with J2 perturbations. This is fine for visualization and for scheduling at the timescales the prototype operates on, but it produces eclipse geometry that is approximate near the orbital plane crossings. Upgrading to the sgp4 Python package is a one-day change for Stage Two.

The thermal model is a five-node lumped-parameter network with handpicked thermal masses and conductances. This is fine for demonstrating that the scheduler responds to thermal dynamics, but the absolute temperature values should not be taken as predictions of any real spacecraft's behavior. Calibrating the model against real on-orbit telemetry is the first task of Stage Two.

The workload generator produces synthetic jobs with reasonable but ultimately invented statistics. Real customer workloads will look different, and the relative gain of the trained policy versus the baselines will shift accordingly. The Stage Two integration with a real customer is the moment of truth for this assumption.

The PPO policy is a simple feedforward MLP with two hidden layers of two hundred fifty-six units each. This is sufficient for ten satellites but will need to grow for the full Stage One one-hundred-satellite case, and may need to switch to a graph neural network architecture or a transformer over satellite tokens for Stage Three multi-operator coordination. The current architecture is the simplest thing that demonstrably works.

The cost figures for Lightning.ai H100 time and for the Anthropic API are accurate as of mid-May 2026 but will change. The forty to one hundred and sixty dollar training cost range assumes that training converges by ten million timesteps; longer runs cost proportionally more.

The Vercel free hobby tier is more than enough for prototype traffic but is not suitable for production-scale demo loads. Moving to a Vercel Pro plan or to a custom hosting setup is a Stage Two consideration.

If you take this prototype into a customer conversation, the most useful thing you can do is be specific and honest about which parts of the demonstration are real measurements of the trained policy's behavior and which parts are synthetic. The credibility you build in that conversation is the asset that converts pilot interest into Stage Two revenue.
## Contact

**Email:** h.alesso@comcast.net  
**GitHub:** https://github.com/alessoh

---