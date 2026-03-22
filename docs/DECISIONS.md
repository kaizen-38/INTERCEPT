# Architecture Decision Records

## 1. Frontend Framework: Next.js + App Router

**Context:** Need a modern React framework for the dashboard.

**Options:**
- Vite + React SPA
- Next.js Pages Router
- Next.js App Router

**Decision:** Next.js App Router (latest stable).

**Why:** App Router gives us server components for static data loading (Instant Demo page works without JS), file-based routing, and built-in API routes as a future escape hatch. The tradeoff is slightly more complexity than Vite, but the built-in SSR/SSG capability is critical for the "Instant Demo must work even if backend is down" requirement. Pages Router was considered but App Router is the current recommended path and better for new projects.

---

## 2. Chart Library: Recharts

**Context:** Need bar charts, area charts, and potentially box plots.

**Options:**
- Recharts (declarative, React-native)
- ECharts (imperative, more chart types)
- Victory (React-native, smaller community)
- D3 directly (maximum flexibility)

**Decision:** Recharts.

**Why:** Recharts is declarative/composable in JSX which matches our React mental model. It handles responsive containers, tooltips, and animations out of the box. ECharts has more chart types (violin/box plots) but requires imperative bridge code and is heavier (500KB). For our use case — bar charts with error bars and area charts — Recharts covers everything we need with less bundle size and simpler code. D3 was rejected because building charts from scratch adds no demo value.

---

## 3. Graph Visualization: Cytoscape.js

**Context:** Need interactive network graph rendering at 50-500 nodes.

**Options:**
- Cytoscape.js
- D3-force
- vis.js
- Sigma.js

**Decision:** Cytoscape.js.

**Why:** Cytoscape.js handles our target size (up to 500 nodes) with smooth performance, has built-in layout algorithms, supports batched style updates (critical for animation), and has mature TypeScript types. D3-force would require building all the interaction/zoom/pan ourselves. vis.js has a larger bundle and weaker typing. Sigma.js is WebGL-based which is overkill for our node counts.

---

## 4. API State Management: TanStack Query

**Context:** Frontend needs to fetch from multiple API endpoints with loading/error states.

**Options:**
- TanStack Query (React Query)
- SWR
- Raw fetch + useState
- RTK Query

**Decision:** TanStack Query.

**Why:** Provides caching, background refetching, loading/error/success states, and mutation support out of the box. The `placeholderData` feature is perfect for our "show static data while fetching from API" pattern on the Instant Demo page. SWR is similar but has weaker mutation support. Raw fetch would require reimplementing all of this. RTK Query requires Redux which is unnecessary overhead.

---

## 5. Backend Framework: FastAPI

**Context:** Need a Python web server that wraps our existing src/ modules.

**Options:**
- FastAPI
- Flask
- Django REST Framework

**Decision:** FastAPI.

**Why:** Python-native (can directly import src/*), automatic OpenAPI spec generation, Pydantic validation baked in, async support, and excellent developer experience with auto-docs at /docs. Flask would work but lacks automatic validation and OpenAPI generation. Django is too heavyweight for an API-only service. FastAPI's OpenAPI spec also enables typed client generation.

---

## 6. API Contract Strategy: Pydantic + Zod

**Context:** Need type safety across the frontend-backend boundary.

**Options:**
- Manual types on both sides
- openapi-typescript codegen
- Pydantic (backend) + Zod (frontend) manual mirror

**Decision:** Pydantic schemas on backend, manually mirrored Zod schemas on frontend, with runtime validation.

**Why:** The Pydantic schemas define the source of truth and produce the OpenAPI spec. On the frontend, Zod schemas provide runtime validation so the UI never renders garbage data from API changes. The manual mirroring is a pragmatic choice — full codegen (openapi-typescript) adds a build step that can break and requires the API to be running during frontend builds. For this project size, the ~10 shared types are easy to keep in sync manually, and both sides get runtime validation.

---

## 7. Log-Prob Fix: forced_delay Parameter

**Context:** Training forces delay=0 but logs log_prob for the sampled delay, breaking policy gradient correctness.

**Options:**
- A) Add `policy.log_prob_of_action()` helper to recompute
- B) Add `forced_delay` parameter to `sample_action()`
- C) Stop forcing delay and add config flag

**Decision:** Option B — `forced_delay` parameter on `sample_action()`.

**Why:** Simplest change with smallest surface area. The fix is localized to one method signature change and one callsite update. Option A requires a separate method that duplicates forward-pass logic. Option C is a larger refactor that changes the training API. With Option B, the forced delay value is used directly in `time_dist.log_prob(delay)`, ensuring the logged log_prob is always consistent with the actual action taken. A unit test verifies this: forced delay=0 log_prob matches recomputed log_prob within 1e-5 tolerance.

---

## 8. Static Fallback for Instant Demo

**Context:** "Instant Demo must work even if backend is down."

**Options:**
- Bundle JSON files into frontend at build time via import
- Fetch from a CDN
- Embed in a Next.js API route

**Decision:** Direct JSON import in the frontend code, used as `placeholderData` in TanStack Query.

**Why:** Zero runtime dependency. The JSON files in `results/` are imported at build time and always available. When the backend is up, TanStack Query fetches fresh data and replaces the static fallback seamlessly. No CDN setup needed, and it works in any deployment environment.

---

## 9. Timing Head Transparency

**Context:** Training forces delay=0, so the timing head may be under-trained.

**Decision:** UI shows explicit warnings on every page where timing is relevant:
1. Demo page: info banner explaining "INTERCEPT (no timing)" vs "INTERCEPT (GRPO)"
2. Sandbox: checkbox for "Force delay=0" with warning when unchecked
3. Both clearly label that timing training must be explicitly enabled

**Why:** Honesty about model behavior is more impressive in an interview than hiding limitations. The ablation toggle in Sandbox lets evaluators explore the difference themselves.
