"use client";

import { type StrategyResult } from "@/lib/api-client";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ErrorBar,
} from "recharts";

const COLORS = [
  "#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899", "#06b6d4", "#84cc16",
];

interface Props {
  data: Record<string, StrategyResult>;
  title: string;
  subtitle?: string;
}

export function BarChartCard({ data, title, subtitle }: Props) {
  const sorted = Object.entries(data).sort(
    ([, a], [, b]) => a.mean_infected - b.mean_infected,
  );

  const chartData = sorted.map(([name, r], i) => ({
    name: name.length > 16 ? name.slice(0, 14) + "…" : name,
    value: r.mean_infected,
    std: r.std_infected,
    fill: COLORS[i % COLORS.length],
  }));

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{title}</CardTitle>
        {subtitle && (
          <p className="text-xs text-zinc-400">{subtitle}</p>
        )}
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={chartData} margin={{ top: 5, right: 20, bottom: 60, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e4e4e7" />
            <XAxis
              dataKey="name"
              angle={-35}
              textAnchor="end"
              tick={{ fontSize: 11 }}
              interval={0}
            />
            <YAxis tick={{ fontSize: 11 }} label={{ value: "Mean Infected", angle: -90, position: "insideLeft", style: { fontSize: 11 } }} />
            <Tooltip
              contentStyle={{
                background: "white",
                border: "1px solid #e4e4e7",
                borderRadius: 8,
                fontSize: 12,
              }}
            />
            <Bar dataKey="value" radius={[4, 4, 0, 0]}>
              {chartData.map((entry, i) => (
                <rect key={i} fill={entry.fill} />
              ))}
              <ErrorBar dataKey="std" width={4} strokeWidth={1.5} stroke="#71717a" />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
