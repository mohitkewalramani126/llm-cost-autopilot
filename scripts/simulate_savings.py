from pathlib import Path

import pandas as pd

from app.registry import MODEL_REGISTRY
from app.router import ROUTING, detect_provider, predict_tier

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "labeled_prompts.csv"

# Rough token estimate since we aren't calling live models here.
# ~1.3 tokens per word is a common approximation for English text.
WORDS_TO_TOKENS = 1.3

# Assumed average output length per tier (tokens) — simple answers are short,
# complex answers are long. These are estimates, not measured values, and
# could be replaced with real averages from data/baseline_results.json later.
AVG_OUTPUT_TOKENS = {"simple": 60, "moderate": 250, "complex": 600}

TIERS = ["simple", "moderate", "complex"]


def get_model(provider, tier):
    model_id = ROUTING[provider][tier]
    for model in MODEL_REGISTRY:
        if model.provider == provider and model.model_id == model_id:
            return model
    raise ValueError(f"No model for {provider}/{tier}")


def estimate_cost(model, input_tokens, output_tokens):
    return input_tokens * model.cost_per_input_token + output_tokens * model.cost_per_output_token


def run_simulation():
    df = pd.read_csv(DATA_PATH)
    provider = detect_provider()
    complex_model = get_model(provider, "complex")

    stats = {tier: {"count": 0, "tokens": 0, "routed_cost": 0.0, "baseline_cost": 0.0} for tier in TIERS}

    for prompt in df["prompt"]:
        predicted_tier = predict_tier(prompt)

        input_tokens = len(prompt.split()) * WORDS_TO_TOKENS
        output_tokens = AVG_OUTPUT_TOKENS[predicted_tier]
        total_tokens = input_tokens + output_tokens

        routed_model = get_model(provider, predicted_tier)
        routed_cost = estimate_cost(routed_model, input_tokens, output_tokens)
        baseline_cost = estimate_cost(complex_model, input_tokens, output_tokens)

        s = stats[predicted_tier]
        s["count"] += 1
        s["tokens"] += total_tokens
        s["routed_cost"] += routed_cost
        s["baseline_cost"] += baseline_cost

    total_prompts = sum(s["count"] for s in stats.values())
    total_tokens = sum(s["tokens"] for s in stats.values())
    total_routed_cost = sum(s["routed_cost"] for s in stats.values())
    total_baseline_cost = sum(s["baseline_cost"] for s in stats.values())
    total_savings = total_baseline_cost - total_routed_cost

    print(f"Provider: {provider}")
    print(f"Prompts simulated: {total_prompts}\n")

    print(f"{'Tier':10s} {'Prompts':>8s} {'% of req':>9s} {'% of tokens':>12s} {'Baseline $':>12s} {'Routed $':>12s} {'Savings':>9s}")
    for tier in TIERS:
        s = stats[tier]
        req_share = 100 * s["count"] / total_prompts if total_prompts else 0
        token_share = 100 * s["tokens"] / total_tokens if total_tokens else 0
        tier_savings_pct = 100 * (s["baseline_cost"] - s["routed_cost"]) / s["baseline_cost"] if s["baseline_cost"] else 0
        print(f"{tier:10s} {s['count']:8d} {req_share:8.1f}% {token_share:11.1f}% {s['baseline_cost']:12.6f} {s['routed_cost']:12.6f} {tier_savings_pct:8.1f}%")

    print()
    print(f"Baseline cost (always {complex_model.model_id}): ${total_baseline_cost:.6f}")
    print(f"Routed cost (tier-based routing):        ${total_routed_cost:.6f}")
    print(f"Overall savings: ${total_savings:.6f} ({100 * total_savings / total_baseline_cost:.1f}%)")


if __name__ == "__main__":
    run_simulation()