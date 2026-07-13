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


if __name__ == "__main__":
    prompt = "What is the capital of France?"
    weak_model = get_model_config("ollama", "simple")
    weak_response = send_request(prompt, weak_model)
    score = judge_response(prompt, weak_response, "ollama")
    print(f"Response: {weak_response.output_text.strip()}")
    print(f"Judge score: {score}/5")