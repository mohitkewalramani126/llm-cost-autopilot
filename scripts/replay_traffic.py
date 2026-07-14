import time

import pandas as pd

from app.pipeline import handle_request

df = pd.read_csv("data/labeled_prompts.csv")
prompts = df["prompt"].tolist()

start = time.perf_counter()
for i, prompt in enumerate(prompts, start=1):
    response = handle_request(prompt)
    print(f"[{i}/{len(prompts)}] {response.output_text.strip()[:80]}")

total = time.perf_counter() - start
print(f"\nDone: {len(prompts)} prompts in {total:.1f}s ({total / len(prompts):.2f}s avg)")

print("Waiting for background verification to finish...")
time.sleep(10)