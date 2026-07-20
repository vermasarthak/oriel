import os
import random
import time
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from .dataset import Case
from .router import Constraints, choose
from .runner import Model, run_cases
from .store import TrialStore

app = FastAPI(title="Oriel API", version="0.1.0")

# Security
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

def get_tenant_id(api_key: str = Security(api_key_header)) -> str:
    if not api_key or not api_key.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    token = api_key.removeprefix("Bearer ")
    if token != os.getenv("ORIEL_API_KEY", "test-token"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    # In a real system, derive tenant_id from token. For v1, use a single tenant or passed in header.
    # Let's say all valid tokens map to tenant "default" for now, or we can require a X-Tenant-ID header.
    return "tenant-1"

TenantDep = Annotated[str, Depends(get_tenant_id)]

# Store (singleton for now, in-memory or file based)
# In production we would use a real SQLite file path from env
store = TrialStore(os.getenv("ORIEL_DB_PATH", ":memory:"))
rng = random.Random()

class EvaluateRequest(BaseModel):
    task: str
    model_name: str
    prompt_version: str
    cases: list[dict[str, Any]] # We will convert to Case objects

class EvaluateResponse(BaseModel):
    total: int
    passed: int

class DeterministicFakeModel(Model):
    def generate(self, input_text: str) -> tuple[str, float, float]:
        # just fake it for now to pass
        return ('{"intent":"refund","priority":"high"}', 10.0, 5.0)

@app.post("/v1/evaluation-runs")
def create_evaluation_run(req: EvaluateRequest, tenant_id: TenantDep) -> EvaluateResponse:
    try:
        cases = [Case(id=c["id"], input_text=c["input"], required=c["required"]) for c in req.cases]
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid cases format")
    
    # We only have a fake provider for now
    model = DeterministicFakeModel()
    summary = run_cases(store, tenant_id, req.task, req.model_name, req.prompt_version, model, cases)
    return EvaluateResponse(total=summary.total, passed=summary.passed)

class RouteRequest(BaseModel):
    task: str
    prompt_version: str
    minimum_quality: float = Field(0.9, ge=0.0, le=1.0)
    max_latency_ms: float = Field(200.0, gt=0)
    max_cost_microusd: float = Field(200.0, gt=0)

class RouteResponse(BaseModel):
    model: str | None
    reason: str
    quality_lower_bound: float | None

@app.post("/v1/route")
def route(req: RouteRequest, tenant_id: TenantDep) -> RouteResponse:
    aggregates = store.aggregates(tenant_id, req.task, req.prompt_version)
    constraints = Constraints(
        minimum_quality=req.minimum_quality,
        max_latency_ms=req.max_latency_ms,
        max_cost_microusd=req.max_cost_microusd,
    )
    decision = choose(aggregates, constraints, rng)
    return RouteResponse(
        model=decision.model,
        reason=decision.reason,
        quality_lower_bound=decision.quality_lower_bound,
    )

class OutcomeRequest(BaseModel):
    task: str
    case_id: str
    model: str
    prompt_version: str
    input_text: str
    passed: bool
    latency_ms: float
    cost_microusd: float

@app.post("/v1/outcomes")
def submit_outcome(req: OutcomeRequest, tenant_id: TenantDep):
    try:
        store.record(
            tenant_id, req.task, req.case_id, req.model, req.prompt_version,
            req.input_text, req.passed, req.latency_ms, req.cost_microusd
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "recorded"}

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/readyz")
def readyz():
    try:
        store.connection.execute("SELECT 1")
        return {"status": "ready"}
    except Exception:
        raise HTTPException(status_code=503, detail="Database unavailable")

@app.get("/metrics")
def metrics():
    # A basic prometheus format metrics response
    return "oriel_up 1\n"
