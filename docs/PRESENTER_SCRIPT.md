# INTERCEPT — Live Demo Presenter Script

**Total time:** ~3 minutes  
**Setup:** Both terminals running (backend on :8000, frontend on :3000). Browser open to http://localhost:3000. Screen shared.

---

## Part 1: The Hook (30 seconds)

**Show:** The Home page at localhost:3000

Start talking as soon as the page is visible:

> "So this is INTERCEPT. The problem we're solving is cascade control on networks — think disease outbreaks, misinformation spreading on social media, or failure propagation in infrastructure. You have a limited budget of interventions, and the question is: who do you protect, and when?"

**Point your cursor** at the three feature cards as you mention each:

> "We built a GNN-based reinforcement learning policy that looks at the actual graph structure — not just simple statistics. We train it with Group Relative Policy Optimization, and we benchmark it against six classical centrality heuristics across multiple network types."

---

## Part 2: The Results (60 seconds)

**Click** the green "Run Instant Demo" button (or the "Instant Demo" nav link).

**Wait** for the page to load. You should see the BA-80 leaderboard and bar chart.

> "These are real evaluation results — no cherry-picking. On a Barabási-Albert network with 80 nodes..."

**Point your cursor** at the leaderboard table, specifically the top row:

> "...our policy variant with forced immediate interventions achieves just 2.8 mean infected nodes. For comparison, PageRank — which is usually the strongest heuristic — gets 9.2. Degree centrality gets 10. Random selection gets 27."

**Point** at the "INTERCEPT (GRPO)" row lower in the table:

> "Now I want to be honest about something. The full policy with the timing head — where it also decides *when* to intervene — scores 20. That's because training currently forces delay to zero, so the timing head is under-trained. We flag this clearly in the UI."

**Point** at the blue info banner about timing head.

**Scroll down** to the Multi-Network section.

> "We didn't just test on one graph. Here are results across six different network topologies — Barabási-Albert, Erdős-Rényi, and Watts-Strogatz, at two scales each."

**Click through 2-3 network tabs** (try BA-100, then ER-100, then WS-100). Pause briefly on each so the audience can see the leaderboard change.

> "You can see our policy wins on four of six networks. On the Erdős-Rényi and Watts-Strogatz graphs, where degree distribution is more uniform, the learned policy actually has an advantage because it can pick up on subtler structural patterns that centrality heuristics miss."

**Scroll down** to the Email-EU section.

> "And here's the real-world test — the SNAP Email-EU network, about a thousand nodes. Every strategy struggles here because the graph is so dense that the cascade is nearly total. We include this because we think showing where a method fails is just as important as showing where it succeeds."

---

## Part 3: The Interactive Sandbox (60 seconds)

**Click** "Sandbox" in the navigation bar.

> "Now let me show you a live simulation."

**The parameter panel** should already have reasonable defaults (BA, 50 nodes, degree strategy). If not, set them.

> "I'm going to run a Barabási-Albert network with 50 nodes, infection probability of 5%, and a budget of 5 interventions using the Degree centrality strategy."

**Click** the green "Run Simulation" button. Wait for the graph to appear.

> "Here's the network. Grey nodes are susceptible, red are infected, and green are protected."

**Click the Play button** to animate. Let it run for a few seconds.

> "Watch how the infection spreads outward from the seed nodes while the strategy protects the highest-degree hubs. You can see the cascade slowing down as it hits protected nodes."

**Pause** the animation partway through. **Point** at the infection chart:

> "This chart shows infected and protected counts over time. You can scrub through the timeline..."

**Drag the slider** back and forth to show the scrubbing.

> "...and see exactly what happened at each step."

**Point** at the Step Details panel:

> "Here you can see which node was protected at each step and the current infection count."

**Now change the strategy** to "Random" in the dropdown and click "Run Simulation" again.

> "If I switch to random protection and run again..."

**Click Play** and let it run to completion.

> "...you can see the infection spreads much further. Random selection doesn't target the structurally important nodes."

---

## Part 4: The Engineering (30 seconds)

**Switch to your code editor or terminal** briefly. Show the project structure if you want.

> "A few things about how this is built. The backend is FastAPI wrapping the existing Python RL codebase — the same training and evaluation code drives the API. The frontend is Next.js with TypeScript strict mode. All API contracts are validated on both sides — Pydantic on the backend, Zod on the frontend."

**Switch back to the browser** and show the Instant Demo page one more time:

> "And that demo page you saw earlier? It works even if the backend goes down. The evaluation data is bundled into the frontend at build time as a static fallback, so the core story is always available."

---

## Part 5: Close (15 seconds)

> "To summarize: we have a GNN policy trained with GRPO that outperforms classical heuristics on most network types, an honest evaluation showing where it works and where it doesn't, and a production-quality dashboard to explore all of it interactively. Happy to take questions."

---

## If Something Goes Wrong

| What happened | What to do |
|---|---|
| Backend crashed mid-demo | Say "Let me switch to the pre-computed results" and go to the Instant Demo page — it works without the backend |
| Simulation is slow | Reduce nodes to 30, say "Let me use a smaller network for speed" |
| Graph looks weird | Say "The layout is auto-generated by spring physics" and move on — it's not the point |
| Someone asks about the timing head | Be direct: "Training currently forces delay to zero, so the timing head isn't properly trained. The ablation toggle in Sandbox lets you explore the difference. Fixing this is a clear next step." |
| Someone asks about scalability | "We've tested up to 500 nodes in the sandbox. For larger graphs the GNN architecture scales but you'd want GPU inference and a smarter layout algorithm." |
| Someone asks why GRPO not PPO | "GRPO eliminates the value function, which simplifies training and avoids the instability of fitting a critic on graph-structured states. The tradeoff is slightly higher variance in advantage estimates, but rank-based normalization keeps it stable." |
