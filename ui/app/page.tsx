"use client";

import { useEffect, useState } from "react";

type Satellite = {
  id: number;
  chip_temp_c: number;
  radiator_temp_c: number;
  battery_soc: number;
  current_load: number;
  eclipse: boolean;
  queued_jobs: number;
  position: number[];
};

type ConstellationState = {
  t_seconds: number;
  satellites: Satellite[];
  pending_jobs: number;
  completed_jobs: number;
  missed_jobs: number;
  reward: number;
  action: number[];
};

const WS_URL =
  typeof process !== "undefined" &&
  process.env &&
  process.env.NEXT_PUBLIC_WS_URL
    ? process.env.NEXT_PUBLIC_WS_URL
    : "ws://localhost:8000/ws";

export default function Home() {
  const [state, setState] = useState<ConstellationState | null>(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const ws = new WebSocket(WS_URL);
    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onmessage = (event) => {
      try {
        setState(JSON.parse(event.data));
      } catch (e) {
        console.error("Failed to parse WS message", e);
      }
    };
    return () => ws.close();
  }, []);

  return (
    <main>
      <header>
        <h1>OrbitalSched</h1>
        <p style={{ color: "#94a3b8", margin: 0 }}>
          Thermal- and orbit-aware inference scheduler, 10-satellite prototype
        </p>
        <div className="status-row">
          <span className={`status-dot ${connected ? "connected" : "disconnected"}`} />
          {connected ? "Connected" : "Disconnected"}
        </div>
      </header>

      {!state ? (
        <p style={{ color: "#94a3b8", marginTop: 24 }}>Waiting for telemetry...</p>
      ) : (
        <div className="layout">
          <section className="panel">
            <h2>Constellation</h2>
            <div className="sat-grid">
              {state.satellites.map((sat) => (
                <SatelliteCard key={sat.id} sat={sat} action={state.action[sat.id]} />
              ))}
            </div>
          </section>

          <section className="panel">
            <h2>Scheduler</h2>
            <Metric label="Simulated time" value={`${(state.t_seconds / 3600).toFixed(1)} h`} />
            <Metric label="Pending jobs" value={state.pending_jobs} />
            <Metric label="Completed jobs" value={state.completed_jobs} />
            <Metric label="Missed jobs" value={state.missed_jobs} />
            <Metric label="Step reward" value={state.reward.toFixed(2)} />
            <Metric
              label="SLA compliance"
              value={`${(
                (state.completed_jobs /
                  Math.max(1, state.completed_jobs + state.missed_jobs)) *
                100
              ).toFixed(1)}%`}
            />
          </section>
        </div>
      )}
    </main>
  );
}

function SatelliteCard({ sat, action }: { sat: Satellite; action: number }) {
  const thermalClass =
    sat.chip_temp_c > 80 ? "hot" : sat.chip_temp_c > 60 ? "warm" : "";
  return (
    <div className={`sat-card ${thermalClass}`}>
      <div className="sat-header">
        <span>SAT-{sat.id.toString().padStart(2, "0")}</span>
        {sat.eclipse ? (
          <span className="eclipse">eclipse</span>
        ) : (
          <span className="sun">sun</span>
        )}
      </div>
      <div>chip: {sat.chip_temp_c.toFixed(1)}°C</div>
      <div>radiator: {sat.radiator_temp_c.toFixed(1)}°C</div>
      <div>battery: {(sat.battery_soc * 100).toFixed(0)}%</div>
      <div>load: {(sat.current_load * 100).toFixed(0)}%</div>
      <div>queue: {sat.queued_jobs}</div>
      <div className="load-bar">
        <div className="load-bar-fill" style={{ width: `${Math.min(100, action * 100)}%` }} />
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="metric-row">
      <span className="metric-label">{label}</span>
      <span className="metric-value">{value}</span>
    </div>
  );
}
