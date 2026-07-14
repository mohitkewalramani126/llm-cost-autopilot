from fastapi import FastAPI
from pydantic import BaseModel

from app.pipeline import handle_request

app = FastAPI(title="LLM Cost Autopilot API")


class CompletionRequest(BaseModel):
    prompt: str


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/v1/completions")
def create_completion(request: CompletionRequest):
    response, provider, tier = handle_request(request.prompt)
    return {
        "output": response.output_text,
        "provider": provider,
        "tier": tier,
        "model": response.model_id,
        "reason": (
            f"Prompt classified as '{tier}' complexity, routed to {provider} "
            f"({response.model_id}) -- the cheapest available model for that tier "
            f"given the API keys currently configured."
        ),
        "cost": response.cost,
        "latency": response.latency,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
    }

from app.registry import MODEL_REGISTRY


@app.get("/v1/models")
def list_models():
    return [
        {
            "provider": m.provider,
            "model_id": m.model_id,
            "cost_per_input_token": m.cost_per_input_token,
            "cost_per_output_token": m.cost_per_output_token,
            "avg_latency": m.avg_latency,
            "quality_tier": m.quality_tier,
        }
        for m in MODEL_REGISTRY
    ]

from app.stats import compute_summary_stats


@app.get("/v1/stats")
def get_stats():
    return compute_summary_stats()

from fastapi import HTTPException
import yaml

from app.router import ROUTING, ROUTING_CONFIG_PATH


class RoutingUpdateRequest(BaseModel):
    provider: str
    tier: str
    model_id: str


@app.put("/v1/routing-config")
def update_routing_config(update: RoutingUpdateRequest):
    valid_tiers = {"simple", "moderate", "complex"}
    if update.tier not in valid_tiers:
        raise HTTPException(status_code=400, detail=f"tier must be one of {sorted(valid_tiers)}")

    if update.provider not in ROUTING:
        raise HTTPException(status_code=400, detail=f"unknown provider '{update.provider}'")

    model_exists = any(
        m.provider == update.provider and m.model_id == update.model_id
        for m in MODEL_REGISTRY
    )
    if not model_exists:
        raise HTTPException(
            status_code=400,
            detail=f"'{update.model_id}' is not in the registry for provider '{update.provider}'",
        )

    # Mutating the dict in place (not reassigning ROUTING) means every
    # module that already imported it -- app.router, app.pipeline -- sees
    # the update immediately, since they all hold a reference to the same
    # dict object.
    ROUTING[update.provider][update.tier] = update.model_id

    # Persist to disk so the change survives a server restart.
    with open(ROUTING_CONFIG_PATH) as f:
        full_config = yaml.safe_load(f)
    full_config["routing"][update.provider][update.tier] = update.model_id
    with open(ROUTING_CONFIG_PATH, "w") as f:
        yaml.safe_dump(full_config, f, default_flow_style=False, sort_keys=False)

    return {"status": "updated", "provider": update.provider, "tier": update.tier, "model_id": update.model_id}