from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from app.features import FEATURE_COLUMNS

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "features.csv"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "tier_classifier.joblib"



TIER_NAMES = {0: "simple", 1: "moderate", 2: "complex"}


def train_classifier():
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLUMNS]
    y = df["tier_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Test accuracy: {accuracy:.3f}\n")
    print(classification_report(y_test, y_pred, target_names=["simple", "moderate", "complex"]))

    print("Feature importances:")
    for name, importance in sorted(
        zip(FEATURE_COLUMNS, model.feature_importances_), key=lambda x: -x[1]
    ):
        print(f"  {name}: {importance:.3f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model -> {MODEL_PATH}")


if __name__ == "__main__":
    train_classifier()