import sys
import time

import pandas as pd

from app.pipeline import handle_request

csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/labeled_prompts.csv"

df = pd.read_csv(csv_path)
prompts = df["prompt"].tolist()

start = time.perf_counter()
for i, prompt in enumerate(prompts, start=1):
    response, provider, tier = handle_request(prompt)
    print(f"[{i}/{len(prompts)}] ({tier}/{provider}) {response.output_text.strip()[:80]}")

total = time.perf_counter() - start
print(f"\nDone: {len(prompts)} prompts from {csv_path} in {total:.1f}s ({total / len(prompts):.2f}s avg)")

print("Waiting for background verification to finish...")
time.sleep(10)