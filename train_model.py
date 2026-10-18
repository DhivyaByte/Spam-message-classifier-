"""Train and evaluate a beginner-friendly SMS spam classifier.

Project description
-------------------
This script downloads the UCI SMS Spam Collection, cleans its labelled text,
splits it into training and test sets, turns text into TF-IDF numeric features,
and trains a Logistic Regression classifier. It prints useful test metrics and
saves the complete scikit-learn pipeline for app.py to use.

Run this file from the project folder:
    python train_model.py

The first run requires an internet connection. The downloaded dataset is not
bundled with this project; see README.md for the dataset citation and license.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib


RANDOM_STATE = 42
TEST_SIZE = 0.20
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "spam_classifier.joblib"
RESULTS_DIR = PROJECT_DIR / "results"


def load_dataset() -> pd.DataFrame:
    """Download the UCI data and return a table with message and label columns."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as error:
        raise SystemExit(
            "The dataset downloader is missing. Install project packages with "
            "`python -m pip install -r requirements.txt`."
        ) from error

    print("Downloading the SMS Spam Collection from UCI...")
    dataset = fetch_ucirepo(id=228)
    features = dataset.data.features
    targets = dataset.data.targets
    if features is None or targets is None or features.empty or targets.empty:
        raise RuntimeError("UCI returned an empty dataset. Please try again later.")

    # The UCI package exposes the message and label as data-frame columns.
    data = pd.DataFrame(
        {
            "message": features.iloc[:, 0].astype("string"),
            "label": targets.iloc[:, 0].astype("string"),
        }
    )
    return data


def clean_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Clean the two useful columns while keeping the original message words."""
    cleaned = data[["message", "label"]].copy()
    cleaned["message"] = cleaned["message"].fillna("").astype(str).str.strip()
    cleaned["label"] = cleaned["label"].fillna("").astype(str).str.strip().str.lower()

    # UCI uses 'ham' for a legitimate, not-spam message and 'spam' for spam.
    cleaned["label"] = cleaned["label"].replace(
        {"not spam": "ham", "legitimate": "ham"}
    )
    cleaned = cleaned[cleaned["message"] != ""]
    cleaned = cleaned[cleaned["label"].isin(["ham", "spam"])]
    cleaned = cleaned.drop_duplicates(subset=["message", "label"]).reset_index(drop=True)

    if cleaned["label"].nunique() != 2:
        raise ValueError("The cleaned data must contain both ham and spam messages.")
    return cleaned


def make_confusion_matrix(matrix: object, output_path: Path) -> None:
    """Save a labeled confusion matrix image for the README/report."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Not spam", "Spam"],
        yticklabels=["Not spam", "Spam"],
        xlabel="Predicted label",
        ylabel="Actual label",
        title="SMS spam classifier — test set",
    )
    threshold = float(max(max(row) for row in matrix)) / 2
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            ax.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=13,
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run the full data preparation, training, evaluation, and save steps."""
    data = clean_dataset(load_dataset())
    messages = data["message"]
    labels = data["label"]

    print(f"Usable messages: {len(data):,}")
    print("Class counts:")
    print(labels.value_counts().rename(index={"ham": "not spam", "spam": "spam"}))

    # Stratification keeps a similar spam/ham ratio in both subsets.
    X_train, X_test, y_train, y_test = train_test_split(
        messages,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    # TF-IDF converts text into numeric word features. Putting it inside the
    # Pipeline means its vocabulary is learned only from the training set.
    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.98,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    print(f"\nTraining on {len(X_train):,} messages...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, list(model.classes_).index("spam")]

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_spam": float(precision_score(y_test, predictions, pos_label="spam")),
        "recall_spam": float(recall_score(y_test, predictions, pos_label="spam")),
        "f1_spam": float(f1_score(y_test, predictions, pos_label="spam")),
        "roc_auc": float(roc_auc_score((y_test == "spam").astype(int), probabilities)),
        "test_size": int(len(y_test)),
        "class_counts": {
            "not_spam": int((labels == "ham").sum()),
            "spam": int((labels == "spam").sum()),
        },
        "random_state": RANDOM_STATE,
    }

    print("\nTest-set metrics (spam is the positive class):")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {name.replace('_', ' ').title():<20} {value:.3f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=["ham", "spam"],
            target_names=["not spam", "spam"],
            zero_division=0,
        )
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    matrix = confusion_matrix(y_test, predictions, labels=["ham", "spam"])
    make_confusion_matrix(matrix, RESULTS_DIR / "confusion_matrix.png")

    print(f"Saved model to: {MODEL_PATH.relative_to(PROJECT_DIR)}")
    print("Saved metrics to: results/metrics.json")
    print("Saved confusion matrix to: results/confusion_matrix.png")


if __name__ == "__main__":
    main()
