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

_classifier = joblib.load(CLASSIFIER_PATH)

with open(ROUTING_CONFIG_PATH) as f:
    _routing_config = yaml.safe_load(f)

ROUTING = _routing_config["routing"]
PROVIDER_PRIORITY = _routing_config["provider_priority"]


def detect_provider() -> str:
    """Pick which provider to route to, based on which API keys are set.
    Falls back to ollama (no key required) if none are available."""
    for provider in PROVIDER_PRIORITY:
        if provider == "ollama":
            continue
        env_var = API_KEY_ENV_VARS.get(provider)
        if env_var and os.getenv(env_var):
            return provider
    return "ollama"


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
    """End-to-end: detect active provider, classify prompt complexity,
    and return the ModelConfig that should handle this prompt."""
    provider = detect_provider()
    tier = predict_tier(prompt)
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