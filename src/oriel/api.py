import os
import random
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, Security, status, BackgroundTasks
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from .dataset import Case
from .router import Constraints, choose
from .runner import Model, run_cases
from .store import TrialStore

try:
    from .providers import OpenAIProvider
    HAS_PROVIDERS = True
except ImportError:
    HAS_PROVIDERS = False

app = FastAPI(title="Oriel API", version="0.1.0")

# Security
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

def get_tenant_id(api_key: str = Security(api_key_header)) -> str:
    if not api_key or not api_key.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    token = api_key.removeprefix("Bearer ")
    if token != os.getenv("ORIEL_API_KEY", "test-token"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return "tenant-1"

TenantDep = Annotated[str, Depends(get_tenant_id)]

store = TrialStore(os.getenv("ORIEL_DB_PATH", ":memory:"))
rng = random.Random()

class EvaluateRequest(BaseModel):
    task: str
    model_name: str
    prompt_version: str
    cases: list[dict[str, Any]]
    use_real_provider: bool = False

class EvaluateResponse(BaseModel):
    total: int
    passed: int

class DeterministicFakeModel(Model):
    def generate(self, input_text: str) -> tuple[str, float, float]:
        return ('{"intent":"refund","priority":"high"}', 10.0, 5.0)

@app.post("/v1/evaluation-runs")
def create_evaluation_run(req: EvaluateRequest, tenant_id: TenantDep, background_tasks: BackgroundTasks) -> EvaluateResponse:
    try:
        cases = [Case(id=c["id"], input_text=c["input"], required=c["required"]) for c in req.cases]
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid cases format")
    
    if req.use_real_provider and HAS_PROVIDERS and os.getenv("OPENAI_API_KEY"):
        model = OpenAIProvider(model_name=req.model_name)
    else:
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

def record_outcome_task(tenant_id: str, req: OutcomeRequest):
    try:
        store.record(
            tenant_id, req.task, req.case_id, req.model, req.prompt_version,
            req.input_text, req.passed, req.latency_ms, req.cost_microusd
        )
    except ValueError:
        pass # Ignore duplicates in async task

@app.post("/v1/outcomes", status_code=status.HTTP_202_ACCEPTED)
def submit_outcome(req: OutcomeRequest, tenant_id: TenantDep, background_tasks: BackgroundTasks):
    # Offload the database write to a background task so the API responds instantly
    background_tasks.add_task(record_outcome_task, tenant_id, req)
    return {"status": "accepted"}

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
    return "oriel_up 1\n"
