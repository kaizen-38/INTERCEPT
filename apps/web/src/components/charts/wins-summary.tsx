"use client";

import { type StrategyResult } from "@/lib/api-client";
import { Badge } from "@/components/ui/badge";
import { Trophy } from "lucide-react";

interface Props {
  multiData: Record<string, Record<string, StrategyResult>>;
}

export function WinsSummary({ multiData }: Props) {
  const wins: Record<string, number> = {};

  for (const strats of Object.values(multiData)) {
    let bestName = "";
    let bestVal = Infinity;
    for (const [name, r] of Object.entries(strats)) {
      if (r.mean_infected < bestVal) {
        bestVal = r.mean_infected;
        bestName = name;
      }
    }
    if (bestName) {
      wins[bestName] = (wins[bestName] ?? 0) + 1;
    }
  }

  const sorted = Object.entries(wins).sort(([, a], [, b]) => b - a);
  const total = Object.keys(multiData).length;

  return (
    <div className="flex flex-wrap gap-3">
      {sorted.map(([name, count]) => (
        <div
          key={name}
          className="flex items-center gap-2 rounded-lg border border-zinc-200 dark:border-zinc-800 px-3 py-2"
        >
          <Trophy className="h-4 w-4 text-amber-500" />
          <span className="text-sm font-medium">{name}</span>
          <Badge variant="secondary">
            {count}/{total} wins
          </Badge>
        </div>
      ))}
    </div>
  );
}
