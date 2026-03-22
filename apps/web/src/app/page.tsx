import Link from "next/link";
import { Shield, BarChart3, FlaskConical, Zap, Network, Brain } from "lucide-react";
import { Card, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";

export default function HomePage() {
  return (
    <div className="flex flex-col gap-16">
      {/* Hero */}
      <section className="flex flex-col items-center text-center pt-12 pb-4 gap-6">
        <div className="flex items-center gap-3 rounded-full border border-emerald-200 bg-emerald-50 px-4 py-1.5 text-sm font-medium text-emerald-700 dark:border-emerald-800 dark:bg-emerald-950 dark:text-emerald-300">
          <Shield className="h-4 w-4" />
          Reinforcement Learning for Epidemic Control
        </div>

        <h1 className="text-5xl font-extrabold tracking-tight leading-tight max-w-3xl">
          Control cascades on networks with{" "}
          <span className="text-emerald-600">limited interventions</span>
        </h1>

        <p className="text-lg text-zinc-500 dark:text-zinc-400 max-w-2xl leading-relaxed">
          INTERCEPT trains a Graph Neural Network policy to decide <strong>who</strong> to
          protect and <strong>when</strong>, outperforming classic centrality heuristics on
          diverse network topologies.
        </p>

        <div className="flex gap-4 pt-2">
          <Link
            href="/demo"
            className="inline-flex items-center gap-2 rounded-lg bg-emerald-600 px-6 py-3 text-sm font-semibold text-white shadow-md hover:bg-emerald-700 transition-colors"
          >
            <Zap className="h-4 w-4" />
            Run Instant Demo
          </Link>
          <Link
            href="/sandbox"
            className="inline-flex items-center gap-2 rounded-lg border border-zinc-300 bg-white px-6 py-3 text-sm font-semibold text-zinc-700 shadow-sm hover:bg-zinc-50 transition-colors dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-200 dark:hover:bg-zinc-800"
          >
            <FlaskConical className="h-4 w-4" />
            Open Sandbox
          </Link>
        </div>
      </section>

      {/* Key features */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card>
          <CardHeader>
            <Network className="h-8 w-8 text-blue-600 mb-2" />
            <CardTitle>Network-Aware</CardTitle>
            <CardDescription>
              GNN encoder captures graph topology — hubs, bridges, and community
              structure all inform intervention decisions.
            </CardDescription>
          </CardHeader>
        </Card>

        <Card>
          <CardHeader>
            <Brain className="h-8 w-8 text-purple-600 mb-2" />
            <CardTitle>Learned Policy vs Heuristics</CardTitle>
            <CardDescription>
              Compare a trained GRPO policy against degree, PageRank, betweenness, closeness,
              k-shell, and random baselines.
            </CardDescription>
          </CardHeader>
        </Card>

        <Card>
          <CardHeader>
            <BarChart3 className="h-8 w-8 text-amber-600 mb-2" />
            <CardTitle>Multi-Network Evaluation</CardTitle>
            <CardDescription>
              Tested across Barabási-Albert, Erdős-Rényi, Watts-Strogatz topologies at
              multiple scales, plus real-world Email-EU data.
            </CardDescription>
          </CardHeader>
        </Card>
      </section>

      {/* How it works */}
      <section className="flex flex-col gap-6">
        <h2 className="text-2xl font-bold tracking-tight">How it works</h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {[
            { step: "1", title: "Cascade starts", desc: "Random seed infections appear in the network." },
            { step: "2", title: "Policy observes", desc: "GNN encodes node states, degrees, and timestep." },
            { step: "3", title: "Intervene", desc: "Policy selects which node to protect and optional delay." },
            { step: "4", title: "Minimize spread", desc: "Objective: fewest infected nodes at episode end." },
          ].map((item) => (
            <div key={item.step} className="flex gap-3 items-start">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-emerald-100 text-sm font-bold text-emerald-700 dark:bg-emerald-900 dark:text-emerald-300">
                {item.step}
              </div>
              <div>
                <p className="font-semibold text-sm">{item.title}</p>
                <p className="text-sm text-zinc-500 dark:text-zinc-400">{item.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
