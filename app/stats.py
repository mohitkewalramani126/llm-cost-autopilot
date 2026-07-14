import sqlite3
from pathlib import Path

import pandas as pd
import yaml

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "requests.db"
MODELS_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "models.yaml"
ROUTING_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "routing.yaml"


def load_requests() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM requests", conn)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["total_cost"] = df["cost"] + df["cost_delta"]
    df["hour"] = df["timestamp"].dt.hour
    return df


def load_pricing() -> dict:
    with open(MODELS_CONFIG_PATH) as f:
        models = yaml.safe_load(f)["models"]
    with open(ROUTING_CONFIG_PATH) as f:
        routing = yaml.safe_load(f)["routing"]
    price = {
        (m["provider"], m["model_id"]): m["cost_per_input_token"] + m["cost_per_output_token"]
        for m in models
    }
    top_model = {provider: tiers["complex"] for provider, tiers in routing.items()}
    return {"price": price, "top_model": top_model}


def estimate_baseline_cost(df: pd.DataFrame, pricing: dict) -> pd.Series:
    """What each request would have cost on the top-tier model for its
    provider. Escalated requests already reflect top-tier pricing exactly;
    everything else is scaled by the per-token price ratio between the
    routed model and the top-tier model, since only a prompt hash is
    logged, not token counts."""
    price, top_model = pricing["price"], pricing["top_model"]

    def row_baseline(row):
        if row["escalated"] or row["tier"] == "complex":
            return row["total_cost"]
        used_price = price.get((row["provider"], row["model_id"]))
        top_price = price.get((row["provider"], top_model.get(row["provider"])))
        if not used_price or not top_price:
            return row["total_cost"]
        return row["total_cost"] * (top_price / used_price)

    return df.apply(row_baseline, axis=1)


def compute_summary_stats() -> dict:
    """Headline cost/savings/escalation numbers, aggregated once so both
    the dashboard and the API report identical figures."""
    df = load_requests()
    if df.empty:
        return {
            "total_requests": 0,
            "total_cost": 0.0,
            "baseline_cost": 0.0,
            "savings": 0.0,
            "savings_pct": 0.0,
            "escalation_rate": 0.0,
            "by_tier": {},
        }

    pricing = load_pricing()
    df["baseline_cost"] = estimate_baseline_cost(df, pricing)

    total_cost = df["total_cost"].sum()
    baseline_cost = df["baseline_cost"].sum()
    savings = baseline_cost - total_cost
    savings_pct = (savings / baseline_cost * 100) if baseline_cost else 0.0
    escalation_rate = (df["escalated"].sum() / len(df) * 100) if len(df) else 0.0

    by_tier = {}
    for tier, group in df.groupby("tier"):
        by_tier[tier] = {
            "count": int(len(group)),
            "total_cost": float(group["total_cost"].sum()),
            "escalation_rate": float(group["escalated"].sum() / len(group) * 100) if len(group) else 0.0,
        }

    return {
        "total_requests": int(len(df)),
        "total_cost": float(total_cost),
        "baseline_cost": float(baseline_cost),
        "savings": float(savings),
        "savings_pct": float(savings_pct),
        "escalation_rate": float(escalation_rate),
        "by_tier": by_tier,
    }