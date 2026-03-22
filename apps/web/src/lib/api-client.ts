/**
 * Typed API client for the INTERCEPT backend.
 * Runtime validation via zod.
 */

import { z } from "zod";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export const StrategyResultSchema = z.object({
  mean_infected: z.number(),
  std_infected: z.number(),
  median_infected: z.number(),
  min_infected: z.number(),
  max_infected: z.number(),
  mean_protected: z.number().optional().nullable(),
});
export type StrategyResult = z.infer<typeof StrategyResultSchema>;

export const ArtifactResponseSchema = z.object({
  data: z.record(z.string(), StrategyResultSchema),
});
export type ArtifactResponse = z.infer<typeof ArtifactResponseSchema>;

export const MultiNetworkResponseSchema = z.object({
  data: z.record(z.string(), z.record(z.string(), StrategyResultSchema)),
});
export type MultiNetworkResponse = z.infer<typeof MultiNetworkResponseSchema>;

const NodePositionSchema = z.object({ x: z.number(), y: z.number() });
const GraphEdgeSchema = z.object({ source: z.number(), target: z.number() });
const StepSnapshotSchema = z.object({
  t: z.number(),
  node_states: z.array(z.number()),
  n_infected: z.number(),
  n_protected: z.number(),
  action_node: z.number().nullable().optional(),
  action_delay: z.number().nullable().optional(),
});

export const SimulationResponseSchema = z.object({
  nodes: z.array(NodePositionSchema),
  edges: z.array(GraphEdgeSchema),
  timeline: z.array(StepSnapshotSchema),
  strategy: z.string(),
  network_info: z.string(),
});
export type SimulationResponse = z.infer<typeof SimulationResponseSchema>;

export const BaselineCompareResponseSchema = z.object({
  results: z.record(z.string(), StrategyResultSchema),
  network_info: z.string(),
});
export type BaselineCompareResponse = z.infer<typeof BaselineCompareResponseSchema>;

export interface SimulationRequest {
  network_type: "ba" | "er" | "ws";
  n_nodes: number;
  infection_prob: number;
  initial_infected: number;
  budget: number;
  max_steps: number;
  strategy: string;
  force_zero_delay: boolean;
  seed?: number | null;
}

export interface BaselineCompareRequest {
  network_type: "ba" | "er" | "ws";
  n_nodes: number;
  infection_prob: number;
  initial_infected: number;
  budget: number;
  max_steps: number;
  n_trials: number;
  seed?: number | null;
}

async function get<T>(path: string, schema: z.ZodType<T>): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`GET ${path} failed: ${res.status}`);
  const json = await res.json();
  return schema.parse(json);
}

async function post<T>(path: string, body: unknown, schema: z.ZodType<T>): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`POST ${path} failed: ${res.status}`);
  const json = await res.json();
  return schema.parse(json);
}

export const api = {
  health: () => fetch(`${API_BASE}/api/health`).then((r) => r.json()),

  artifacts: {
    evaluation: () => get("/api/artifacts/evaluation", ArtifactResponseSchema),
    multiNetwork: () => get("/api/artifacts/multi-network", MultiNetworkResponseSchema),
    emailEu: () => get("/api/artifacts/email-eu", ArtifactResponseSchema),
  },

  simulate: (req: SimulationRequest) =>
    post("/api/simulate", req, SimulationResponseSchema),

  compareBaselines: (req: BaselineCompareRequest) =>
    post("/api/compare/baselines", req, BaselineCompareResponseSchema),
};
