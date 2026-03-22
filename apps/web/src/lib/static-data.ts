/**
 * Static fallback data bundled at build time (Instant Demo works without API).
 */

import type { ArtifactResponse, MultiNetworkResponse } from "./api-client";

import evaluationRaw from "../data/comparison_results.json";
import multiNetworkRaw from "../data/all_results.json";
import emailEuRaw from "../data/email_eu_results.json";

function wrapArtifact(raw: Record<string, Record<string, unknown>>): ArtifactResponse {
  const data: ArtifactResponse["data"] = {};
  for (const [k, v] of Object.entries(raw)) {
    data[k] = {
      mean_infected: (v as Record<string, number>).mean_infected,
      std_infected: (v as Record<string, number>).std_infected,
      median_infected: (v as Record<string, number>).median_infected,
      min_infected: (v as Record<string, number>).min_infected,
      max_infected: (v as Record<string, number>).max_infected,
      mean_protected: (v as Record<string, number>).mean_protected ?? null,
    };
  }
  return { data };
}

function wrapMulti(
  raw: Record<string, Record<string, Record<string, unknown>>>,
): MultiNetworkResponse {
  const data: MultiNetworkResponse["data"] = {};
  for (const [net, strats] of Object.entries(raw)) {
    data[net] = {};
    for (const [k, v] of Object.entries(strats)) {
      data[net][k] = {
        mean_infected: (v as Record<string, number>).mean_infected,
        std_infected: (v as Record<string, number>).std_infected,
        median_infected: (v as Record<string, number>).median_infected,
        min_infected: (v as Record<string, number>).min_infected,
        max_infected: (v as Record<string, number>).max_infected,
        mean_protected: (v as Record<string, number>).mean_protected ?? null,
      };
    }
  }
  return { data };
}

export const staticEvaluation = wrapArtifact(
  evaluationRaw as Record<string, Record<string, unknown>>,
);
export const staticMultiNetwork = wrapMulti(
  multiNetworkRaw as Record<string, Record<string, Record<string, unknown>>>,
);
export const staticEmailEu = wrapArtifact(
  emailEuRaw as Record<string, Record<string, unknown>>,
);
