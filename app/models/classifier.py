from __future__ import annotations

import os
import pickle
import statistics
from dataclasses import dataclass
from typing import List, Sequence

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from pathlib import Path

LABELS = ["Ekonomi", "Sosial", "Lingkungan", "Administrasi"]  # contoh label
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "storage" / "impact_classifier.pkl"


@dataclass
class PredictResult:
    label: str
    proba: float


def build_demo_model() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000)),
            ("clf", RandomForestClassifier(n_estimators=150, random_state=42)),
        ]
    )


def fit_and_save(train_texts: Sequence[str], train_labels: Sequence[str], path: str = MODEL_PATH):
    if not train_texts:
        raise ValueError("Dataset pelatihan kosong.")
    model = build_demo_model()
    model.fit(train_texts, train_labels)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(model, handle)


def load_model(path: str = MODEL_PATH) -> Pipeline | None:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as handle:
        return pickle.load(handle)


def predict(text: str, model: Pipeline) -> PredictResult:
    proba = model.predict_proba([text])[0]
    idx = int(proba.argmax())
    label = LABELS[idx] if idx < len(LABELS) else str(idx)
    return PredictResult(label=label, proba=float(proba[idx]))


def evaluate_with_cross_validation(
    texts: Sequence[str], labels: Sequence[str], folds: int = 3
) -> dict:
    if folds < 2:
        raise ValueError("Jumlah fold minimal 2.")
    if len(texts) < folds:
        raise ValueError("Data tidak cukup untuk cross-validation.")

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    f1_scores: List[float] = []

    for train_index, test_index in cv.split(texts, labels):
        model = build_demo_model()
        train_texts = [texts[i] for i in train_index]
        train_labels = [labels[i] for i in train_index]
        model.fit(train_texts, train_labels)
        test_texts = [texts[i] for i in test_index]
        test_labels = [labels[i] for i in test_index]
        predictions = model.predict(test_texts)
        report = classification_report(test_labels, predictions, output_dict=True, zero_division=0)
        f1_scores.append(report["weighted avg"]["f1-score"])

    mean_f1 = statistics.mean(f1_scores)
    std_f1 = statistics.pstdev(f1_scores) if len(f1_scores) > 1 else 0.0
    return {
        "folds": folds,
        "f1_weighted_mean": round(mean_f1, 4),
        "f1_weighted_std": round(std_f1, 4),
    }