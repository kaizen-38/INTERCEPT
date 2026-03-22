"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
} from "recharts";

interface TimelinePoint {
  t: number;
  n_infected: number;
  n_protected: number;
}

interface Props {
  timeline: TimelinePoint[];
  maxStep: number;
}

export function InfectionChart({ timeline, maxStep }: Props) {
  const data = timeline.slice(0, maxStep + 1).map((p) => ({
    t: p.t,
    Infected: p.n_infected,
    Protected: p.n_protected,
    Susceptible:
      (timeline[0] ? timeline[0].n_infected + (timeline[0].n_protected ?? 0) : 0) > 0
        ? undefined
        : undefined,
  }));

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Infection Over Time</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={220}>
          <AreaChart data={data} margin={{ top: 5, right: 10, bottom: 5, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e4e4e7" />
            <XAxis dataKey="t" tick={{ fontSize: 11 }} label={{ value: "Step", position: "insideBottom", offset: -2, style: { fontSize: 11 } }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip contentStyle={{ background: "white", border: "1px solid #e4e4e7", borderRadius: 8, fontSize: 12 }} />
            <Area type="monotone" dataKey="Infected" stroke="#ef4444" fill="#fecaca" strokeWidth={2} />
            <Area type="monotone" dataKey="Protected" stroke="#10b981" fill="#d1fae5" strokeWidth={2} />
          </AreaChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
