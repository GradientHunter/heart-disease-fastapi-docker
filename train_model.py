from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

DATA_PATH = Path("heart.csv")
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "heart_model.joblib"

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
TARGET = "target"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "heart.csv was not found. Download the Heart Disease Dataset "
            "from the assignment's Kaggle link and place heart.csv in the project root."
        )

    df = pd.read_csv(DATA_PATH)
    required_columns = FEATURES + [TARGET]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            "Expected the JohnSmith88 dataset columns and target."
        )

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X_train, y_train)

    accuracy = classifier.score(X_test, y_test)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {
            "model": classifier,
            "model_type": "RandomForestClassifier",
            "features": FEATURES,
        },
        MODEL_PATH,
    )
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Hold-out accuracy (informational only): {accuracy:.4f}")


if __name__ == "__main__":
    main()
