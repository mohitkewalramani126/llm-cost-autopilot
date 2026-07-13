import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.models import ModelConfig, Response
from app.verifier import VerificationResult

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "requests.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    provider TEXT NOT NULL,
    tier TEXT NOT NULL,
    model_id TEXT NOT NULL,
    cost REAL NOT NULL,
    latency REAL NOT NULL,
    verified INTEGER NOT NULL,
    judge_score INTEGER,
    escalated INTEGER NOT NULL,
    cost_delta REAL NOT NULL
)
"""


def _init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(_SCHEMA)


def log_request(
    prompt: str,
    provider: str,
    tier: str,
    routed_model: ModelConfig,
    response: Response,
    verification: VerificationResult,
) -> None:
    """Persist one request's routing + verification outcome to SQLite.
    Stores a prompt hash rather than the raw prompt text."""
    _init_db()
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO requests
                (timestamp, prompt_hash, provider, tier, model_id, cost, latency,
                 verified, judge_score, escalated, cost_delta)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                prompt_hash,
                provider,
                tier,
                routed_model.model_id,
                response.cost,
                response.latency,
                int(verification.verified),
                verification.score,
                int(verification.escalated),
                verification.cost_delta,
            ),
        )


if __name__ == "__main__":
    from app.client import send_request
    from app.router import get_model_config
    from app.verifier import verify_and_escalate

    prompt = "Prove that the square root of 2 is irrational, step by step."
    provider = "ollama"
    tier = "simple"

    routed_model = get_model_config(provider, tier)
    response = send_request(prompt, routed_model)
    verification = verify_and_escalate(prompt, response, provider, tier, force=True)

    log_request(prompt, provider, tier, routed_model, response, verification)

    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT * FROM requests ORDER BY id DESC LIMIT 1").fetchone()
    print(row)