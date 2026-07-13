import sqlite3
from pathlib import Path

import pandas as pd

from app.logger import DB_PATH
from scripts.extract_features import run_feature_extraction
from scripts.train_classifier import train_classifier

LABELED_PATH = Path(__file__).resolve().parent.parent / "data" / "labeled_prompts.csv"


def run_flywheel():
    with sqlite3.connect(DB_PATH) as conn:
        candidates = pd.read_sql(
            "SELECT id, prompt, tier_corrected FROM retraining_candidates WHERE incorporated = 0",
            conn,
        )

        if candidates.empty:
            print("No new escalations to incorporate. Skipping retrain.")
            return

        new_rows = candidates.rename(columns={"tier_corrected": "tier"})[["prompt", "tier"]]
        existing = pd.read_csv(LABELED_PATH)
        combined = pd.concat([existing, new_rows], ignore_index=True).drop_duplicates(subset="prompt")
        combined.to_csv(LABELED_PATH, index=False)
        print(f"Added {len(new_rows)} escalated prompts to {LABELED_PATH} ({len(combined)} total)")

        placeholders = ",".join("?" * len(candidates))
        conn.execute(
            f"UPDATE retraining_candidates SET incorporated = 1 WHERE id IN ({placeholders})",
            candidates["id"].tolist(),
        )
        conn.commit()

    print("Retraining classifier on updated dataset...")
    run_feature_extraction()
    train_classifier()


if __name__ == "__main__":
    run_flywheel()