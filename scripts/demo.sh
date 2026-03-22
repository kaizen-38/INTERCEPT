#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

echo "============================================="
echo "  INTERCEPT — Demo Launcher"
echo "============================================="
echo ""

# ── Python deps ──────────────────────────────────────────────────────
echo "[1/4] Installing Python dependencies..."
if [ -f requirements.txt ]; then
    pip install -q -r requirements.txt 2>/dev/null || pip3 install -q -r requirements.txt 2>/dev/null || true
fi
pip install -q fastapi uvicorn pydantic sse-starlette 2>/dev/null || pip3 install -q fastapi uvicorn pydantic sse-starlette 2>/dev/null || true

# ── Frontend deps ────────────────────────────────────────────────────
echo "[2/4] Installing frontend dependencies..."
if command -v pnpm &>/dev/null; then
    (cd apps/web && pnpm install --frozen-lockfile 2>/dev/null || pnpm install)
elif command -v npm &>/dev/null; then
    (cd apps/web && npm install)
else
    echo "ERROR: No package manager found. Install pnpm or npm."
    exit 1
fi

# ── Start backend ────────────────────────────────────────────────────
echo "[3/4] Starting API server on :8000..."
PYTHONPATH="$REPO_ROOT" python -m uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!
echo "  API PID: $API_PID"

# Wait for backend to be ready
for i in $(seq 1 15); do
    if curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
        echo "  API is ready."
        break
    fi
    sleep 1
done

# ── Start frontend ───────────────────────────────────────────────────
echo "[4/4] Starting frontend on :3000..."
(cd apps/web && pnpm dev --port 3000 2>/dev/null || npm run dev -- --port 3000) &
WEB_PID=$!
echo "  Web PID: $WEB_PID"

echo ""
echo "============================================="
echo "  INTERCEPT is running!"
echo ""
echo "  Dashboard:  http://localhost:3000"
echo "  API docs:   http://localhost:8000/docs"
echo ""
echo "  → Click 'Instant Demo' to see results"
echo "  → Click 'Sandbox' for live simulation"
echo ""
echo "  Press Ctrl+C to stop all services."
echo "============================================="

cleanup() {
    echo ""
    echo "Shutting down..."
    kill $API_PID 2>/dev/null || true
    kill $WEB_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

wait
