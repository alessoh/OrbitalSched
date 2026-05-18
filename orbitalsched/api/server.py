"""
FastAPI server exposing the OrbitalSched scheduler to the UI.

Run with:
    uvicorn orbitalsched.api.server:app --host 0.0.0.0 --port 8000

The server loads the trained policy if available, otherwise falls back
to a constant action policy for development. The WebSocket endpoint at
/ws streams live constellation state to the Vercel-hosted UI at 2 Hz.
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
        print(f"No checkpoint at {CHECKPOINT_PATH}; using constant policy.")
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
                assert state.env.dc is not None
                state.obs, _ = state.env.reset(
                    seed=int(state.env.dc.t_seconds) % 1000
                )
            await asyncio.sleep(0.5)  # 2 Hz live update
    except WebSocketDisconnect:
        pass
