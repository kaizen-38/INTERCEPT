# INTERCEPT Demo Script

**Duration:** 90 seconds  
**Prerequisite:** `make demo` or `./scripts/demo.sh` running

---

## Narration + Click Path

### 1. Home Page (10 sec)

> "INTERCEPT is a reinforcement learning system for controlling cascades on networks. Think disease outbreaks, misinformation spread, or network failures. The core challenge: you have a **limited budget** of interventions — who do you protect, and when?"

- Page loads to hero section
- Point out the 3 feature cards

### 2. Instant Demo (40 sec)

> "Let me show you the results."

- Click **"Run Instant Demo"** button

> "This page shows pre-computed evaluation results — no GPU, no training needed."

**BA-80 Leaderboard:**

> "On a Barabási-Albert network with 80 nodes, we compared 6 centrality heuristics against our GNN policy. The 'INTERCEPT (no timing)' variant — which forces immediate interventions — achieves just 2.8 mean infected nodes, beating PageRank at 9.2 and Degree at 10.0."

- Point to the leaderboard table and bar chart

**Multi-Network:**

> "We tested across 6 network types: BA, ER, and Watts-Strogatz at two scales each. Click through the tabs — you'll see our policy wins on 4 of 6 networks."

- Click through BA-100, ER-100, WS-100 tabs
- Point out the Wins Summary badges

**Email-EU:**

> "On a real-world email network with 1,005 nodes, every strategy struggles — the cascade is nearly total. This is an honest evaluation: we show where the method works and where it doesn't."

### 3. Sandbox (30 sec)

> "Now let's run a live simulation."

- Click **"Open Sandbox"** in nav

> "I'll set up a small BA network with 50 nodes, infection probability 0.05, budget of 5."

- Defaults are already set
- Click **"Run Simulation"**

> "Watch the graph animate — red nodes are infected, green are protected, grey are susceptible. The Degree strategy protects hub nodes first."

- Click Play to animate
- Point to the infection-over-time chart
- Point to the step details panel

> "You can try different strategies, network types, and parameters. The 'Force delay = 0' toggle controls whether interventions are immediate or use the policy's timing head."

### 4. Wrap-up (10 sec)

> "The system is built with FastAPI + Next.js, typed contracts end-to-end, and the instant demo works even if the backend goes down — it falls back to bundled data. Questions?"

---

## If Something Breaks

| Issue | Fallback |
|-------|----------|
| Backend won't start | Instant Demo still works — uses bundled JSON |
| Sandbox simulation fails | Check `uvicorn` is running, check terminal for Python errors |
| Frontend won't build | Run `pnpm -C apps/web build` to see TypeScript errors |
| Charts look wrong | Hard-refresh browser (Cmd+Shift+R) |

## Quick Recovery Commands

```bash
# Restart backend
cd /path/to/INTERCEPT
python -m uvicorn apps.api.main:app --port 8000

# Restart frontend
cd apps/web
pnpm dev

# Full restart
./scripts/demo.sh
```
