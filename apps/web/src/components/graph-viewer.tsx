"use client";

import { useEffect, useRef, useCallback } from "react";
import cytoscape, { type Core } from "cytoscape";
import type { SimulationResponse } from "@/lib/api-client";

interface Props {
  sim: SimulationResponse;
  currentStep: number;
}

const STATE_COLORS: Record<number, string> = {
  0: "#94a3b8", // susceptible — slate
  1: "#ef4444", // infected — red
  2: "#10b981", // protected — emerald
};

export function GraphViewer({ sim, currentStep }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);

  const initGraph = useCallback(() => {
    if (!containerRef.current) return;

    cyRef.current?.destroy();

    const elements = [
      ...sim.nodes.map((n, i) => ({
        data: { id: String(i) },
        position: { x: n.x * 400 + 300, y: n.y * 400 + 300 },
      })),
      ...sim.edges.map((e, i) => ({
        data: { id: `e${i}`, source: String(e.source), target: String(e.target) },
      })),
    ];

    cyRef.current = cytoscape({
      container: containerRef.current,
      elements,
      style: [
        {
          selector: "node",
          style: {
            width: 12,
            height: 12,
            "background-color": "#94a3b8",
            "border-width": 0,
            label: "",
          },
        },
        {
          selector: "edge",
          style: {
            width: 0.5,
            "line-color": "#e2e8f0",
            "curve-style": "bezier",
          },
        },
        {
          selector: "node.action-target",
          style: {
            "border-width": 3,
            "border-color": "#f59e0b",
            width: 18,
            height: 18,
          },
        },
      ],
      layout: { name: "preset" },
      userZoomingEnabled: true,
      userPanningEnabled: true,
      boxSelectionEnabled: false,
    });
  }, [sim]);

  useEffect(() => {
    initGraph();
    return () => {
      cyRef.current?.destroy();
    };
  }, [initGraph]);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;

    const snapshot = sim.timeline[currentStep];
    if (!snapshot) return;

    cy.batch(() => {
      cy.nodes().removeClass("action-target");

      snapshot.node_states.forEach((state, i) => {
        const node = cy.getElementById(String(i));
        node.style("background-color", STATE_COLORS[state] ?? "#94a3b8");
      });

      if (snapshot.action_node != null) {
        cy.getElementById(String(snapshot.action_node)).addClass("action-target");
      }
    });
  }, [sim, currentStep]);

  return (
    <div className="relative rounded-xl border border-zinc-200 dark:border-zinc-800 overflow-hidden bg-zinc-50 dark:bg-zinc-900">
      <div ref={containerRef} className="w-full" style={{ height: 400 }} />
      <div className="absolute bottom-3 left-3 flex gap-3 text-xs">
        {[
          { label: "Susceptible", color: "#94a3b8" },
          { label: "Infected", color: "#ef4444" },
          { label: "Protected", color: "#10b981" },
        ].map(({ label, color }) => (
          <div key={label} className="flex items-center gap-1.5">
            <div className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: color }} />
            <span className="text-zinc-600 dark:text-zinc-400">{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
