"use client";

import { type StrategyResult } from "@/lib/api-client";
import { Badge } from "@/components/ui/badge";
import { Trophy } from "lucide-react";

interface Props {
  data: Record<string, StrategyResult>;
  title?: string;
}

export function LeaderboardTable({ data, title }: Props) {
  const sorted = Object.entries(data).sort(
    ([, a], [, b]) => a.mean_infected - b.mean_infected,
  );

  return (
    <div className="overflow-x-auto">
      {title && (
        <h3 className="text-sm font-semibold text-zinc-500 dark:text-zinc-400 mb-3 uppercase tracking-wider">
          {title}
        </h3>
      )}
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-zinc-200 dark:border-zinc-800 text-left">
            <th className="pb-2 pr-4 font-medium text-zinc-500">#</th>
            <th className="pb-2 pr-4 font-medium text-zinc-500">Strategy</th>
            <th className="pb-2 pr-4 font-medium text-zinc-500 text-right">Mean Infected</th>
            <th className="pb-2 pr-4 font-medium text-zinc-500 text-right">Std</th>
            <th className="pb-2 pr-4 font-medium text-zinc-500 text-right">Median</th>
            <th className="pb-2 pr-4 font-medium text-zinc-500 text-right">Min</th>
            <th className="pb-2 pr-4 font-medium text-zinc-500 text-right">Max</th>
            <th className="pb-2 font-medium text-zinc-500 text-right">Protected</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map(([name, r], i) => (
            <tr
              key={name}
              className="border-b border-zinc-100 dark:border-zinc-800/50 hover:bg-zinc-50 dark:hover:bg-zinc-900/50 transition-colors"
            >
              <td className="py-2.5 pr-4">
                {i === 0 ? (
                  <Trophy className="h-4 w-4 text-amber-500" />
                ) : (
                  <span className="text-zinc-400">{i + 1}</span>
                )}
              </td>
              <td className="py-2.5 pr-4 font-medium">
                {name}
                {i === 0 && (
                  <Badge variant="success" className="ml-2">
                    Best
                  </Badge>
                )}
              </td>
              <td className="py-2.5 pr-4 text-right font-mono tabular-nums">
                {r.mean_infected.toFixed(1)}
              </td>
              <td className="py-2.5 pr-4 text-right font-mono tabular-nums text-zinc-400">
                {r.std_infected.toFixed(1)}
              </td>
              <td className="py-2.5 pr-4 text-right font-mono tabular-nums">
                {r.median_infected.toFixed(0)}
              </td>
              <td className="py-2.5 pr-4 text-right font-mono tabular-nums text-zinc-400">
                {r.min_infected}
              </td>
              <td className="py-2.5 pr-4 text-right font-mono tabular-nums text-zinc-400">
                {r.max_infected}
              </td>
              <td className="py-2.5 text-right font-mono tabular-nums">
                {r.mean_protected != null ? r.mean_protected.toFixed(1) : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
