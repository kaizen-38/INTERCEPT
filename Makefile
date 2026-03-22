.PHONY: demo api web install test lint

demo:
	./scripts/demo.sh

api:
	PYTHONPATH=. python3 -m uvicorn apps.api.main:app --reload --port 8000

web:
	cd apps/web && pnpm dev

install:
	pip3 install -r requirements.txt
	pip3 install fastapi uvicorn pydantic sse-starlette
	cd apps/web && pnpm install

test:
	PYTHONPATH=. python3 -m pytest tests/ -v
	cd apps/web && pnpm lint

lint:
	ruff check src/ apps/api/ tests/ || true
	cd apps/web && pnpm lint
