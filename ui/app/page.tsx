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

// In Next.js builds, process.env.NEXT_PUBLIC_WS_URL is inlined at build time.
// In a standalone browser preview (e.g. Claude artifact viewer), `process` is
// not defined, so we guard the access and fall back to localhost.
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
    <main className="min-h-screen bg-slate-950 text-slate-100 p-6">
      <header className="mb-6">
        <h1 className="text-3xl font-bold">OrbitalSched</h1>
        <p className="text-slate-400">
          Thermal- and orbit-aware inference scheduler, 10-satellite prototype
        </p>
        <div className="mt-2 inline-flex items-center gap-2 text-sm">
          <span
            className={`inline-block w-2 h-2 rounded-full ${
              connected ? "bg-emerald-500" : "bg-rose-500"
            }`}
          />
          {connected ? "Connected" : "Disconnected"}
        </div>
      </header>

      {!state ? (
        <p className="text-slate-400">Waiting for telemetry...</p>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <section className="lg:col-span-2 bg-slate-900 rounded-lg p-4">
            <h2 className="text-lg font-semibold mb-3">Constellation</h2>
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
              {state.satellites.map((sat) => (
                <SatelliteCard key={sat.id} sat={sat} action={state.action[sat.id]} />
              ))}
            </div>
          </section>

          <section className="bg-slate-900 rounded-lg p-4">
            <h2 className="text-lg font-semibold mb-3">Scheduler</h2>
            <Metric
              label="Simulated time"
              value={`${(state.t_seconds / 3600).toFixed(1)} h`}
            />
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
  const thermalColor =
    sat.chip_temp_c > 80
      ? "border-rose-500"
      : sat.chip_temp_c > 60
      ? "border-amber-500"
      : "border-emerald-600";
  return (
    <div className={`rounded-md border ${thermalColor} bg-slate-800 p-2 text-xs`}>
      <div className="flex justify-between items-center mb-1">
        <span className="font-semibold">SAT-{sat.id.toString().padStart(2, "0")}</span>
        {sat.eclipse ? (
          <span className="text-indigo-300">eclipse</span>
        ) : (
          <span className="text-yellow-300">sun</span>
        )}
      </div>
      <div>chip: {sat.chip_temp_c.toFixed(1)}°C</div>
      <div>radiator: {sat.radiator_temp_c.toFixed(1)}°C</div>
      <div>battery: {(sat.battery_soc * 100).toFixed(0)}%</div>
      <div>load: {(sat.current_load * 100).toFixed(0)}%</div>
      <div>queue: {sat.queued_jobs}</div>
      <div className="mt-1 h-1 bg-slate-700 rounded">
        <div
          className="h-1 bg-sky-500 rounded"
          style={{ width: `${Math.min(100, action * 100)}%` }}
        />
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex justify-between py-1 border-b border-slate-800 last:border-0">
      <span className="text-slate-400">{label}</span>
      <span className="font-mono">{value}</span>
    </div>
  );
}
