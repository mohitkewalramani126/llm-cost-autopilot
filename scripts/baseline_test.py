import json
import time
from pathlib import Path

from app.registry import MODEL_REGISTRY
from app.client import send_request, MissingAPIKeyError

PROMPTS = [
    "Extract the name and email from this text: 'Contact John Smith at john@example.com for details.'",
    "What is the capital of France?",
    "Reformat this list into bullet points: apples, bananas, oranges, grapes",
    "Summarize this in one sentence: The quarterly report showed a 12% increase in revenue, driven mainly by strong performance in the European market, while North American sales remained flat.",
    "Classify the sentiment of this review as positive, negative, or neutral: 'The product works fine but shipping took forever.'",
    "Compare the pros and cons of remote work versus office work.",
    "Write a short creative story about a robot learning to paint.",
    "Analyze the following scenario and suggest three possible causes: website traffic dropped 40% overnight with no code deployments.",
    "Given these constraints — budget under $500, must fit in a small apartment, needs a webcam — recommend a laptop setup.",
    "Explain the tradeoffs between SQL and NoSQL databases for a startup building a social media app.",
]

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "baseline_results.json"


def run_baseline():
    results = []

    for model in MODEL_REGISTRY:
        for prompt in PROMPTS:
            print(f"Sending to {model.model_id}: {prompt[:50]}...")
            try:
                response = send_request(prompt, model)
            except MissingAPIKeyError as e:
                print(f"  Skipped: {e}")
                continue
            except Exception as e:
                print(f"  Error: {e}")
                continue

            results.append({
                "provider": model.provider,
                "model_id": model.model_id,
                "quality_tier": model.quality_tier,
                "prompt": prompt,
                "output_text": response.output_text,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "latency": response.latency,
                "cost": response.cost,
            })

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} results to {OUTPUT_PATH}")


if __name__ == "__main__":
    run_baseline()