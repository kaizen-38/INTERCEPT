"use client";

import { useState, useCallback } from "react";
import { useMutation } from "@tanstack/react-query";
import { api, type SimulationRequest } from "@/lib/api-client";
import { GraphViewer } from "@/components/graph-viewer";
import { InfectionChart } from "@/components/charts/infection-chart";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Play, SkipForward, SkipBack, Pause, AlertTriangle, Info } from "lucide-react";
import { useEffect, useRef } from "react";

const STRATEGIES = [
  { value: "degree", label: "Degree Centrality" },
  { value: "pagerank", label: "PageRank" },
  { value: "betweenness", label: "Betweenness" },
  { value: "closeness", label: "Closeness" },
  { value: "kshell", label: "K-Shell" },
  { value: "random", label: "Random" },
] as const;

const NETWORK_TYPES = [
  { value: "ba" as const, label: "Barabási-Albert" },
  { value: "er" as const, label: "Erdős-Rényi" },
  { value: "ws" as const, label: "Watts-Strogatz" },
] as const;

function SelectField({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: readonly { value: string; label: string }[];
}) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-xs font-medium text-zinc-500 dark:text-zinc-400">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-md border border-zinc-300 bg-white px-2.5 py-1.5 text-sm dark:border-zinc-700 dark:bg-zinc-900"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
}

function NumberField({
  label,
  value,
  onChange,
  min,
  max,
  step,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
}) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-xs font-medium text-zinc-500 dark:text-zinc-400">{label}</span>
      <input
        type="number"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        min={min}
        max={max}
        step={step}
        className="rounded-md border border-zinc-300 bg-white px-2.5 py-1.5 text-sm tabular-nums dark:border-zinc-700 dark:bg-zinc-900"
      />
    </label>
  );
}

export default function SandboxPage() {
  const [networkType, setNetworkType] = useState<"ba" | "er" | "ws">("ba");
  const [nNodes, setNNodes] = useState(50);
  const [infectionProb, setInfectionProb] = useState(0.05);
  const [initialInfected, setInitialInfected] = useState(3);
  const [budget, setBudget] = useState(5);
  const [maxSteps, setMaxSteps] = useState(30);
  const [strategy, setStrategy] = useState("degree");
  const [forceZeroDelay, setForceZeroDelay] = useState(true);

  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const mutation = useMutation({
    mutationFn: (req: SimulationRequest) => api.simulate(req),
    onSuccess: () => {
      setCurrentStep(0);
      setIsPlaying(false);
    },
  });

  const sim = mutation.data;
  const maxTimelineStep = sim ? sim.timeline.length - 1 : 0;

  const handleRun = useCallback(() => {
    mutation.mutate({
      network_type: networkType,
      n_nodes: nNodes,
      infection_prob: infectionProb,
      initial_infected: initialInfected,
      budget,
      max_steps: maxSteps,
      strategy,
      force_zero_delay: forceZeroDelay,
      seed: 42,
    });
  }, [networkType, nNodes, infectionProb, initialInfected, budget, maxSteps, strategy, forceZeroDelay, mutation]);

  // Playback
  useEffect(() => {
    if (isPlaying && sim) {
      intervalRef.current = setInterval(() => {
        setCurrentStep((prev) => {
          if (prev >= maxTimelineStep) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 300);
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isPlaying, sim, maxTimelineStep]);

  const currentSnapshot = sim?.timeline[currentStep];

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Sandbox</h1>
        <p className="text-zinc-500 dark:text-zinc-400 mt-1">
          Run interactive simulations and watch cascade dynamics in real time.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Controls */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle className="text-base">Parameters</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            <SelectField
              label="Network"
              value={networkType}
              onChange={(v) => setNetworkType(v as "ba" | "er" | "ws")}
              options={NETWORK_TYPES}
            />
            <NumberField label="Nodes" value={nNodes} onChange={setNNodes} min={10} max={500} />
            <NumberField
              label="Infection Prob"
              value={infectionProb}
              onChange={setInfectionProb}
              min={0.01}
              max={1}
              step={0.01}
            />
            <NumberField
              label="Initial Infected"
              value={initialInfected}
              onChange={setInitialInfected}
              min={1}
              max={50}
            />
            <NumberField label="Budget" value={budget} onChange={setBudget} min={1} max={50} />
            <NumberField label="Max Steps" value={maxSteps} onChange={setMaxSteps} min={5} max={200} />
            <SelectField
              label="Strategy"
              value={strategy}
              onChange={setStrategy}
              options={STRATEGIES}
            />

            <label className="flex items-center gap-2 pt-1">
              <input
                type="checkbox"
                checked={forceZeroDelay}
                onChange={(e) => setForceZeroDelay(e.target.checked)}
                className="rounded border-zinc-300"
              />
              <span className="text-xs text-zinc-600 dark:text-zinc-400">Force delay = 0</span>
            </label>

            {!forceZeroDelay && (
              <div className="flex items-start gap-1.5 text-xs text-amber-700 dark:text-amber-300 bg-amber-50 dark:bg-amber-950 border border-amber-200 dark:border-amber-800 rounded-md px-2.5 py-2">
                <Info className="h-3 w-3 shrink-0 mt-0.5" />
                Timing head may be under-trained unless training enables it.
              </div>
            )}

            <button
              onClick={handleRun}
              disabled={mutation.isPending}
              className="mt-2 flex items-center justify-center gap-2 rounded-lg bg-emerald-600 px-4 py-2 text-sm font-semibold text-white hover:bg-emerald-700 transition-colors disabled:opacity-50"
            >
              <Play className="h-4 w-4" />
              {mutation.isPending ? "Simulating…" : "Run Simulation"}
            </button>

            {mutation.isError && (
              <div className="flex items-center gap-2 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700 dark:border-red-800 dark:bg-red-950 dark:text-red-300">
                <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
                {mutation.error instanceof Error ? mutation.error.message : "Simulation failed"}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Visualization area */}
        <div className="lg:col-span-3 flex flex-col gap-4">
          {!sim && !mutation.isPending && (
            <div className="flex items-center justify-center rounded-xl border border-dashed border-zinc-300 dark:border-zinc-700 h-[400px] text-zinc-400">
              Configure parameters and click &quot;Run Simulation&quot;
            </div>
          )}

          {mutation.isPending && (
            <div className="flex items-center justify-center rounded-xl border border-dashed border-zinc-300 dark:border-zinc-700 h-[400px]">
              <div className="flex items-center gap-3 text-zinc-500">
                <div className="h-5 w-5 animate-spin rounded-full border-2 border-zinc-300 border-t-emerald-600" />
                Running simulation…
              </div>
            </div>
          )}

          {sim && (
            <>
              <GraphViewer sim={sim} currentStep={currentStep} />

              {/* Playback controls */}
              <div className="flex items-center gap-3">
                <button
                  onClick={() => setCurrentStep(0)}
                  className="rounded-md border border-zinc-300 p-1.5 hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-800"
                >
                  <SkipBack className="h-4 w-4" />
                </button>
                <button
                  onClick={() => setIsPlaying(!isPlaying)}
                  className="rounded-md bg-zinc-900 p-1.5 text-white hover:bg-zinc-800 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-200"
                >
                  {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                </button>
                <button
                  onClick={() => setCurrentStep(maxTimelineStep)}
                  className="rounded-md border border-zinc-300 p-1.5 hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-800"
                >
                  <SkipForward className="h-4 w-4" />
                </button>

                <input
                  type="range"
                  min={0}
                  max={maxTimelineStep}
                  value={currentStep}
                  onChange={(e) => {
                    setIsPlaying(false);
                    setCurrentStep(Number(e.target.value));
                  }}
                  className="flex-1"
                />

                <span className="text-sm font-mono tabular-nums text-zinc-500 w-20 text-right">
                  Step {currentStep}/{maxTimelineStep}
                </span>
              </div>

              {/* Stats & chart */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <InfectionChart timeline={sim.timeline} maxStep={currentStep} />

                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-base">Step Details</CardTitle>
                    <CardDescription>{sim.network_info}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    {currentSnapshot && (
                      <div className="grid grid-cols-2 gap-3 text-sm">
                        <div>
                          <span className="text-zinc-500 dark:text-zinc-400">Infected</span>
                          <p className="text-xl font-bold text-red-600 tabular-nums">
                            {currentSnapshot.n_infected}
                          </p>
                        </div>
                        <div>
                          <span className="text-zinc-500 dark:text-zinc-400">Protected</span>
                          <p className="text-xl font-bold text-emerald-600 tabular-nums">
                            {currentSnapshot.n_protected}
                          </p>
                        </div>
                        {currentSnapshot.action_node != null && (
                          <div className="col-span-2">
                            <span className="text-zinc-500 dark:text-zinc-400">Action</span>
                            <div className="font-medium">
                              Protect node{" "}
                              <Badge variant="secondary">{currentSnapshot.action_node}</Badge>
                              {currentSnapshot.action_delay != null && (
                                <span className="ml-2 text-zinc-400">
                                  delay={currentSnapshot.action_delay}
                                </span>
                              )}
                            </div>
                          </div>
                        )}
                        <div className="col-span-2">
                          <span className="text-zinc-500 dark:text-zinc-400">Strategy</span>
                          <p className="font-medium capitalize">{sim.strategy}</p>
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
