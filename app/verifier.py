import re

from app.client import send_request
from app.models import Response
from app.router import get_model_config

JUDGE_PROMPT_TEMPLATE = """You are grading an AI assistant's response for correctness and completeness.

Original request:
{prompt}

Assistant's response:
{response}

On a scale of 1 to 5, how well does the response answer the request?
1 = wrong or unusable, 3 = partially correct or incomplete, 5 = fully correct and complete.

Reply in exactly this format, with no other text:
SCORE: <a single integer from 1 to 5>
"""


def _parse_score(judge_text: str) -> int:
    """Pull the integer score out of the judge model's reply. Falls back to
    the first standalone digit 1-5 anywhere if the strict format isn't followed."""
    match = re.search(r"SCORE:\s*(\d)", judge_text)
    if match:
        return int(match.group(1))
    fallback = re.search(r"\b([1-5])\b", judge_text)
    if fallback:
        return int(fallback.group(1))
    raise ValueError(f"Could not parse a score from judge response: {judge_text!r}")


def judge_response(prompt: str, response_to_judge: Response, provider: str) -> int:
    """Send the original prompt and a lower-tier model's response to the
    top-tier model for this provider, and get back a 1-5 quality score."""
    judge_model = get_model_config(provider, "complex")
    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        prompt=prompt, response=response_to_judge.output_text
    )
    judge_reply = send_request(judge_prompt, judge_model)
    return _parse_score(judge_reply.output_text)


import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

VERIFICATION_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "verification.yaml"

with open(VERIFICATION_CONFIG_PATH) as f:
    _verification_config = yaml.safe_load(f)

SAMPLE_RATE = _verification_config["sample_rate"]
SCORE_THRESHOLD = _verification_config["score_threshold"]


@dataclass
class VerificationResult:
    verified: bool             # whether this request was sampled for verification at all
    score: Optional[int]       # judge score, None if not verified
    escalated: bool            # whether we re-ran with the top-tier model
    final_response: Response   # the response the caller should actually use
    cost_delta: float          # extra cost paid due to escalation (0 if none)


def should_verify(tier: str) -> bool:
    """Complex-tier requests are already on the top model, nothing to verify against.
    Everything else gets sampled at SAMPLE_RATE."""
    if tier == "complex":
        return False
    return random.random() < SAMPLE_RATE


def verify_and_escalate(
    prompt: str, response: Response, provider: str, tier: str, force: bool = False
) -> VerificationResult:
    """Judge a routed response; if it's sampled (or forced) and fails the
    quality bar, re-run the prompt on the top-tier model and use that instead."""
    if not (force or should_verify(tier)):
        return VerificationResult(
            verified=False, score=None, escalated=False,
            final_response=response, cost_delta=0.0,
        )

    score = judge_response(prompt, response, provider)

    if score >= SCORE_THRESHOLD:
        return VerificationResult(
            verified=True, score=score, escalated=False,
            final_response=response, cost_delta=0.0,
        )

    top_tier_model = get_model_config(provider, "complex")
    escalated_response = send_request(prompt, top_tier_model)
    cost_delta = escalated_response.cost - response.cost

    return VerificationResult(
        verified=True, score=score, escalated=True,
        final_response=escalated_response, cost_delta=cost_delta,
    )


if __name__ == "__main__":
    prompt = "Prove that the square root of 2 is irrational, step by step."
    weak_model = get_model_config("ollama", "simple")
    weak_response = send_request(prompt, weak_model)
    result = verify_and_escalate(prompt, weak_response, "ollama", "simple", force=True)

    print(f"Judge score: {result.score}/5")
    print(f"Escalated: {result.escalated}")
    print(f"Cost delta: ${result.cost_delta:.6f}")
    print(f"Final response: {result.final_response.output_text.strip()[:200]}")