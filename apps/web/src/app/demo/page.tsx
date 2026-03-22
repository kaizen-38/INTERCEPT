"use client";

import { useQuery } from "@tanstack/react-query";
import { api, type ArtifactResponse, type MultiNetworkResponse } from "@/lib/api-client";
import { staticEvaluation, staticMultiNetwork, staticEmailEu } from "@/lib/static-data";
import { LeaderboardTable } from "@/components/charts/leaderboard-table";
import { BarChartCard } from "@/components/charts/bar-chart-card";
import { WinsSummary } from "@/components/charts/wins-summary";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { AlertTriangle, Info } from "lucide-react";
import { useState } from "react";

function useArtifactWithFallback<T>(
  key: string,
  fetcher: () => Promise<T>,
  fallback: T,
) {
  return useQuery({
    queryKey: [key],
    queryFn: fetcher,
    initialData: fallback,
    retry: false,
    staleTime: 0,
  });
}

export default function DemoPage() {
  const evalQ = useArtifactWithFallback("eval-artifact", api.artifacts.evaluation, staticEvaluation);
  const multiQ = useArtifactWithFallback("multi-artifact", api.artifacts.multiNetwork, staticMultiNetwork);
  const emailQ = useArtifactWithFallback("email-artifact", api.artifacts.emailEu, staticEmailEu);

  const [selectedNetwork, setSelectedNetwork] = useState<string | null>(null);

  const evalData = evalQ.data as ArtifactResponse | undefined;
  const multiData = multiQ.data as MultiNetworkResponse | undefined;
  const emailData = emailQ.data as ArtifactResponse | undefined;

  const isOffline = evalQ.isError || multiQ.isError;

  const networkNames = multiData ? Object.keys(multiData.data) : [];
  const activeNetwork = selectedNetwork ?? networkNames[0] ?? null;

  return (
    <div className="flex flex-col gap-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Instant Demo</h1>
        <p className="text-zinc-500 dark:text-zinc-400 mt-1">
          Pre-computed evaluation results — no training required.
        </p>
        {isOffline && (
          <div className="mt-3 flex items-center gap-2 rounded-lg border border-amber-200 bg-amber-50 px-4 py-2 text-sm text-amber-800 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-200">
            <AlertTriangle className="h-4 w-4 shrink-0" />
            Backend unavailable. Showing bundled static data.
          </div>
        )}
      </div>

      {/* ── Evaluation on BA-80 ──────────────────────────────────── */}
      <section className="flex flex-col gap-4">
        <h2 className="text-xl font-semibold">BA-80 Evaluation</h2>
        <p className="text-sm text-zinc-500 dark:text-zinc-400">
          Baselines vs INTERCEPT policy on Barabási-Albert(80, 3) graph, p=0.05, budget=10.
        </p>

        {evalData && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Leaderboard</CardTitle>
              </CardHeader>
              <CardContent>
                <LeaderboardTable data={evalData.data} />
              </CardContent>
            </Card>
            <BarChartCard
              data={evalData.data}
              title="Mean Infected (lower is better)"
              subtitle="BA(80, 3) · 100 trials · p=0.05"
            />
          </div>
        )}

        <div className="flex items-start gap-2 rounded-lg border border-blue-200 bg-blue-50 px-4 py-3 text-sm text-blue-800 dark:border-blue-800 dark:bg-blue-950 dark:text-blue-200">
          <Info className="h-4 w-4 shrink-0 mt-0.5" />
          <div>
            <strong>Note on timing head:</strong> Training currently forces delay=0, so the
            &quot;INTERCEPT (GRPO)&quot; result uses the full (potentially under-trained) timing head,
            while &quot;INTERCEPT (no timing)&quot; forces delay=0 at inference — which often performs
            better until timing training is enabled.
          </div>
        </div>
      </section>

      {/* ── Multi-Network Comparison ─────────────────────────────── */}
      <section className="flex flex-col gap-4">
        <h2 className="text-xl font-semibold">Multi-Network Comparison</h2>
        <p className="text-sm text-zinc-500 dark:text-zinc-400">
          Performance across BA, ER, and WS topologies at multiple scales.
        </p>

        {multiData && (
          <>
            <WinsSummary multiData={multiData.data} />

            <div className="flex flex-wrap gap-2">
              {networkNames.map((net) => (
                <button
                  key={net}
                  onClick={() => setSelectedNetwork(net)}
                  className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                    net === activeNetwork
                      ? "bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900"
                      : "bg-zinc-100 text-zinc-700 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700"
                  }`}
                >
                  {net}
                </button>
              ))}
            </div>

            {activeNetwork && multiData.data[activeNetwork] && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">{activeNetwork} Leaderboard</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <LeaderboardTable data={multiData.data[activeNetwork]} />
                  </CardContent>
                </Card>
                <BarChartCard
                  data={multiData.data[activeNetwork]}
                  title={`${activeNetwork} — Mean Infected`}
                />
              </div>
            )}
          </>
        )}
      </section>

      {/* ── Email-EU Evaluation ──────────────────────────────────── */}
      <section className="flex flex-col gap-4">
        <h2 className="text-xl font-semibold">Real-World: Email-EU</h2>
        <p className="text-sm text-zinc-500 dark:text-zinc-400">
          SNAP Email-Eu-core network (1,005 nodes, 25,571 edges). A dense real-world graph
          where all strategies struggle — cascade is nearly total.
        </p>

        {emailData && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Email-EU Leaderboard</CardTitle>
              </CardHeader>
              <CardContent>
                <LeaderboardTable data={emailData.data} />
              </CardContent>
            </Card>
            <BarChartCard
              data={emailData.data}
              title="Email-EU — Mean Infected"
              subtitle="1,005 nodes · budget=20"
            />
          </div>
        )}
      </section>
    </div>
  );
}
