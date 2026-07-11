import re
from pathlib import Path

import pandas as pd

INPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "labeled_prompts.csv"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "features.csv"

TIER_LABEL = {"simple": 0, "moderate": 1, "complex": 2}

# Words that tend to signal a multi-step or open-ended task rather than a lookup.
INSTRUCTION_VERBS = [
    "explain", "write", "design", "build", "create", "analyze", "compare",
    "describe", "calculate", "implement", "develop", "propose", "specify",
    "include", "state", "define", "review", "diagnose", "translate",
    "summarize", "draft", "plan", "construct",
]

# Tokens that suggest the prompt itself contains code or a technical artifact.
CODE_TOKENS = ["def ", "class ", "SELECT ", "function", "{", "}", ";", "import ", "()", "=>"]


def extract_features(prompt: str) -> dict:
    words = prompt.split()
    sentences = [s for s in re.split(r"[.!?]", prompt) if s.strip()]

    word_count = len(words)
    char_count = len(prompt)
    sentence_count = max(len(sentences), 1)
    avg_word_length = round(char_count / word_count, 2) if word_count else 0
    has_question_mark = int("?" in prompt)

    lower_prompt = prompt.lower()
    code_token_count = sum(prompt.count(tok) for tok in CODE_TOKENS)
    instruction_verb_count = sum(1 for verb in INSTRUCTION_VERBS if verb in lower_prompt)

    return {
        "word_count": word_count,
        "char_count": char_count,
        "sentence_count": sentence_count,
        "avg_word_length": avg_word_length,
        "has_question_mark": has_question_mark,
        "code_token_count": code_token_count,
        "instruction_verb_count": instruction_verb_count,
    }


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