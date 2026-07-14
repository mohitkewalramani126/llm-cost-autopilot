import os
from pathlib import Path

import joblib
import pandas as pd
import yaml
from dotenv import load_dotenv

from app.features import extract_features, FEATURE_COLUMNS
from app.models import ModelConfig
from app.registry import MODEL_REGISTRY

load_dotenv()

ROUTING_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "routing.yaml"
CLASSIFIER_PATH = Path(__file__).resolve().parent.parent / "models" / "tier_classifier.joblib"

TIER_NAMES = {0: "simple", 1: "moderate", 2: "complex"}

API_KEY_ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}

# Real key prefixes for each provider. Used to filter out empty values and
# unedited placeholders (e.g. copying .env.example without replacing
# "your-openai-key-here") so the router doesn't mistake a placeholder for a
# usable key and try to route real traffic to it.
API_KEY_PREFIXES = {
    "anthropic": "sk-ant-",
    "openai": "sk-",
}

_classifier = joblib.load(CLASSIFIER_PATH)

with open(ROUTING_CONFIG_PATH) as f:
    _routing_config = yaml.safe_load(f)

ROUTING = _routing_config["routing"]


def _available_paid_providers() -> list:
    """Paid providers that currently have what looks like a real API key
    set -- present, non-empty, and matching that provider's known key
    prefix. This doesn't guarantee the key is valid (only a live API call
    can confirm that), but it does rule out blank values and copy-pasted
    placeholder text."""
    available = []
    for provider, env_var in API_KEY_ENV_VARS.items():
        key = os.getenv(env_var, "").strip()
        if key and key.startswith(API_KEY_PREFIXES[provider]):
            available.append(provider)
    return available


def _combined_price(model: ModelConfig) -> float:
    """Per-token price used only to rank models against each other for the
    same tier -- the same combined input+output price approach the
    dashboard already uses for its baseline-cost comparison. Not a real
    per-request cost estimate on its own."""
    return model.cost_per_input_token + model.cost_per_output_token


def detect_provider(tier: str) -> str:
    """Picks which provider should handle a request for a given tier.

    If zero paid providers have a usable key set, falls back to free local
    Ollama. If exactly one paid provider has a usable key set, uses it. If
    BOTH OpenAI and Anthropic have usable keys set, compares their price for
    THIS SPECIFIC TIER and returns whichever is cheaper -- so different
    tiers can genuinely end up on different providers if that's what the
    real pricing favors, rather than always committing to one provider."""
    paid_providers = _available_paid_providers()
    if not paid_providers:
        return "ollama"
    if len(paid_providers) == 1:
        return paid_providers[0]
    return min(paid_providers, key=lambda p: _combined_price(get_model_config(p, tier)))


def predict_tier(prompt: str) -> str:
    features = extract_features(prompt)
    row = pd.DataFrame([features])[FEATURE_COLUMNS]
    prediction = _classifier.predict(row)[0]
    return TIER_NAMES[prediction]


def get_model_config(provider: str, tier: str) -> ModelConfig:
    model_id = ROUTING[provider][tier]
    for model in MODEL_REGISTRY:
        if model.provider == provider and model.model_id == model_id:
            return model
    raise ValueError(f"No model in registry matches provider={provider} model_id={model_id}")


def route(prompt: str) -> ModelConfig:
    """End-to-end: classify prompt complexity first, then find whichever
    available provider is cheapest for that specific tier, and return the
    ModelConfig that should handle this prompt."""
    tier = predict_tier(prompt)
    provider = detect_provider(tier)
    return get_model_config(provider, tier)


if __name__ == "__main__":
    test_prompts = [
        "What is the capital of France?",
        "Write a Python function that reverses a linked list and explain the time complexity.",
        "Design a fault-tolerant distributed system for processing financial transactions across three regions with exactly-once delivery guarantees.",
    ]
    for p in test_prompts:
        model = route(p)
        print(f"[{model.quality_tier:8s}] provider={model.provider:10s} model={model.model_id:20s} <- {p[:60]}")