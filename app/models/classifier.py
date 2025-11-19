from __future__ import annotations

import json
import logging
import os
import pickle
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

LABELS = ["Ekonomi", "Sosial", "Lingkungan", "Administrasi"]  # contoh label
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "storage" / "impact_classifier.pkl"
DATASET_PATH = BASE_DIR / "models" / "impact_training_samples.json"

logger = logging.getLogger(__name__)


@dataclass
class PredictResult:
    label: str
    proba: float


def build_random_forest() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000)),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]
    )


def build_svm() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000)),
            ("clf", SVC(kernel="linear", probability=True, class_weight="balanced", random_state=42)),
        ]
    )


def _train_with_grid_search(texts: Sequence[str], labels: Sequence[str]) -> Tuple[Pipeline, dict]:
    class_counts = Counter(labels)
    min_class = min(class_counts.values()) if class_counts else 0
    if min_class < 2:
        raise ValueError("Dataset memerlukan minimal dua contoh per kelas untuk optimasi.")
    cv_folds = min(3, min_class)

    candidates = [
        (
            "random_forest",
            build_random_forest(),
            {"clf__n_estimators": [200, 300], "clf__max_depth": [None, 20]},
        ),
        (
            "svm",
            build_svm(),
            {"clf__C": [0.5, 1.0, 2.0]},
        ),
    ]

    best_score = -1.0
    best_model: Pipeline | None = None
    best_metadata: dict = {}

    for name, pipeline, param_grid in candidates:
        grid = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=cv_folds,
            scoring="f1_weighted",
            n_jobs=-1,
        )
        grid.fit(texts, labels)
        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_model = grid.best_estimator_
            best_metadata = {"model": name, "best_score": grid.best_score_, "params": grid.best_params_}

    if best_model is None:
        raise RuntimeError("Grid search tidak menghasilkan model terbaik.")
    return best_model, best_metadata


def fit_and_save(
    train_texts: Sequence[str],
    train_labels: Sequence[str],
    path: str = MODEL_PATH,
    optimize: bool = True,
) -> dict | None:
    if not train_texts:
        raise ValueError("Dataset pelatihan kosong.")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    try:
        if optimize and len(set(train_labels)) > 1:
            model, diagnostics = _train_with_grid_search(train_texts, train_labels)
        else:
            model = build_random_forest()
            model.fit(train_texts, train_labels)
            diagnostics = None
    except ValueError as exc:
        logger.warning("Optimasi model gagal (%s), fallback ke random forest standar.", exc)
        model = build_random_forest()
        model.fit(train_texts, train_labels)
        diagnostics = None

    with open(path, "wb") as handle:
        pickle.dump(model, handle)
    return diagnostics


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
        model = build_random_forest()
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


def load_training_dataset(path: Path = DATASET_PATH) -> Tuple[List[str], List[str]]:
    if not path.exists():
        return [], []
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    texts: List[str] = []
    labels: List[str] = []
    for row in payload:
        text = (row or {}).get("text")
        label = (row or {}).get("label")
        if text and label:
            texts.append(text)
            labels.append(label)
    return texts, labels


def ensure_default_model() -> Pipeline | None:
    model = load_model()
    if model:
        return model
    texts, labels = load_training_dataset()
    if not texts:
        logger.warning("Default training dataset tidak ditemukan. Klasifikasi dinonaktifkan sementara.")
        return None
    fit_and_save(texts, labels)
    return load_model()
