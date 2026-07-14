import threading

from app.client import send_request
from app.logger import log_request
from app.models import Response
from app.router import detect_provider, get_model_config, predict_tier
from app.verifier import verify_and_escalate
from app.logger import log_request, log_retraining_candidate


def _verify_in_background(prompt: str, provider: str, tier: str, routed_model, response: Response) -> None:
    """Runs off the main thread so verification never delays the response
    the caller already has."""
    verification = verify_and_escalate(prompt, response, provider, tier)
    log_request(prompt, provider, tier, routed_model, response, verification)
    if verification.escalated:
        # Escalation always jumps straight to the top tier, so that's the
        # corrected label — the cheap tier was wrong, complex was right.
        log_retraining_candidate(prompt, tier_corrected="complex")


def handle_request(prompt: str) -> Response:
    """The full pipeline: route, get a response, return it immediately,
    and kick off quality verification + logging in the background."""
    tier = predict_tier(prompt)
    provider = detect_provider(tier)
    routed_model = get_model_config(provider, tier)
    response = send_request(prompt, routed_model)

    thread = threading.Thread(
        target=_verify_in_background,
        args=(prompt, provider, tier, routed_model, response),
        daemon=True,
    )
    thread.start()

    return response


if __name__ == "__main__":
    import sqlite3
    import time

    from app.logger import DB_PATH

    with sqlite3.connect(DB_PATH) as conn:
        before = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]

    start = time.perf_counter()
    for i in range(10):
        response = handle_request(f"What is {i} plus {i}?")
        print(f"Got response in {response.latency:.2f}s: {response.output_text.strip()[:40]}")
    total = time.perf_counter() - start
    print(f"All 10 requests returned in {total:.2f}s total")

    time.sleep(5)  # give background verification threads a chance to finish

    with sqlite3.connect(DB_PATH) as conn:
        after = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
    print(f"Verification rows logged: {after - before} (out of 10 requests, sample_rate=0.15)")