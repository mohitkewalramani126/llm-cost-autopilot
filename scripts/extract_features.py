from pathlib import Path

import pandas as pd

from app.features import extract_features

INPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "labeled_prompts.csv"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "features.csv"

TIER_LABEL = {"simple": 0, "moderate": 1, "complex": 2}


def run_feature_extraction():
    df = pd.read_csv(INPUT_PATH)

    feature_rows = df["prompt"].apply(extract_features).apply(pd.Series)
    df = pd.concat([df, feature_rows], axis=1)
    df["tier_label"] = df["tier"].map(TIER_LABEL)

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Extracted features for {len(df)} prompts -> {OUTPUT_PATH}")
    print(df.groupby("tier")[["word_count", "sentence_count", "instruction_verb_count"]].mean())


if __name__ == "__main__":
    run_feature_extraction()