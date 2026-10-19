"""
train_model.py
----------------
Step 2 of the pipeline.

Loads dataset.csv (produced by collect_data.py), trains a Random Forest
classifier on the hand-landmark features, evaluates it on a held-out
test split, and saves the trained model to sign_model.pkl.

Random Forest is used because hand-landmark coordinates are a small,
structured, tabular feature set (63 numbers per sample) rather than
raw image pixels -- tree-based models handle this kind of feature set
very effectively and train in seconds on a laptop CPU, with no GPU
required.

HOW TO USE
----------
    python train_model.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

DATA_FILE = "dataset.csv"
MODEL_FILE = "sign_model.pkl"


def main():
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} samples across {df['label'].nunique()} classes: "
          f"{sorted(df['label'].unique())}")

    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")


if __name__ == "__main__":
    main()
