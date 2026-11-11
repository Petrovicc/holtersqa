from __future__ import annotations

import json
import functools
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from ekg_monitor.infrastructure.butqdb_repository import ButqdbRepository, interval_label

from .models import DatasetBuildConfig, TrainingConfig
from .signal_analysis import apply_filter, extract_feature_map


ProgressCallback = Callable[[float, str], None]
META_COLUMNS = {"record_id", "subject_id", "i0", "i1", "label"}


@dataclass(slots=True)
class TrainingResult:
    estimator: Any
    feature_columns: list[str]
    metrics: dict[str, Any]
    confusion_matrix: np.ndarray
    classes: list[int]
    loss_curve: list[float]
    accuracy_curve: list[float]
    config: TrainingConfig


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """Serializable NumPy inference form of a CUDA-trained PyTorch MLP."""

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...],
        learning_rate_init: float,
        backend: str,
    ) -> None:
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.learning_rate_init = float(learning_rate_init)
        self.backend = str(backend)
        self.activation = "relu"
        self.classes_: np.ndarray = np.asarray([], dtype=int)
        self.coefs_: list[np.ndarray] = []
        self.intercepts_: list[np.ndarray] = []
        self.n_features_in_: int = 0

    def fit(self, matrix: np.ndarray, labels: np.ndarray) -> "TorchMLPClassifier":
        raise RuntimeError(
            "TorchMLPClassifier is fitted by the CUDA training backend"
        )

    def __sklearn_is_fitted__(self) -> bool:
        return bool(self.coefs_ and self.intercepts_ and len(self.classes_))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "learning_rate_init": self.learning_rate_init,
            "activation": self.activation,
            "backend": self.backend,
        }

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=np.float64)
        for index, (weights, biases) in enumerate(
            zip(self.coefs_, self.intercepts_)
        ):
            values = values @ weights + biases
            if index < len(self.coefs_) - 1:
                values = np.maximum(values, 0.0)
        values -= np.max(values, axis=1, keepdims=True)
        probabilities = np.exp(values)
        totals = np.sum(probabilities, axis=1, keepdims=True)
        return probabilities / np.maximum(totals, np.finfo(float).tiny)

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        indices = np.argmax(self.predict_proba(matrix), axis=1)
        return self.classes_[indices]


@functools.lru_cache(maxsize=1)
def cuda_device_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except Exception:
        return None
    return None


def build_dataset(
    repository: ButqdbRepository,
    record_ids: list[str],
    config: DatasetBuildConfig,
    progress: ProgressCallback | None = None,
    cancel: threading.Event | None = None,
) -> pd.DataFrame:
    build_started = __import__("time").perf_counter()
    if not record_ids:
        raise ValueError("Select at least one record")
    if not any(
        (
            config.feature_config.use_ecg_time,
            config.feature_config.use_ecg_spectrum,
            config.feature_config.use_acceleration,
            config.feature_config.use_acceleration_diff,
            config.feature_config.use_demographics,
        )
    ):
        raise ValueError("Select at least one feature group")
    if config.window_seconds <= 0:
        raise ValueError("Window length must be positive")
    if not 0 <= config.overlap <= 0.95:
        raise ValueError("Overlap must be between 0 and 0.95")
    rows: list[dict[str, Any]] = []
    eligible_windows = 0
    capped_records: dict[str, tuple[int, int]] = {}
    for record_index, record_id in enumerate(record_ids):
        _check_cancel(cancel)
        record = repository.get(record_id)
        header = record.ecg_header
        if header is None or record.annotation_path is None:
            continue
        window_samples = int(round(config.window_seconds * header.fs))
        hop = max(1, int(round(window_samples * (1.0 - config.overlap))))
        groups = repository.annotation_groups(record_id)
        base_intervals = _base_intervals(groups, config.label_source)
        candidates: list[tuple[int, int, int]] = []
        seen: set[tuple[int, int]] = set()
        for interval_start, interval_end, _ in base_intervals:
            cursor = interval_start
            while cursor + window_samples <= interval_end:
                window_end = cursor + window_samples
                label = interval_label(groups, cursor, window_end, config.label_source)
                key = (cursor, window_end)
                if label and key not in seen:
                    candidates.append((cursor, window_end, label))
                    seen.add(key)
                cursor += hop
        candidates.sort()
        candidate_count = len(candidates)
        eligible_windows += candidate_count
        maximum = int(config.max_windows_per_record)
        if maximum > 0 and candidate_count > maximum:
            indices = np.linspace(0, len(candidates) - 1, maximum).round().astype(int)
            candidates = [candidates[index] for index in np.unique(indices)]
            capped_records[record_id] = (candidate_count, len(candidates))
        blocks = _group_windows(candidates, header.fs, config.window_seconds)
        total_blocks = max(1, len(blocks))
        for block_index, block in enumerate(blocks):
            _check_cancel(cancel)
            block_start = min(window[0] for window in block)
            block_end = max(window[1] for window in block)
            signals = repository.read_ecg_sample_window(record_id, block_start, block_end)
            ecg_data = signals["ecg"]
            acc_data = signals.get("acc")
            for start, end, label in block:
                e0, e1 = start - block_start, end - block_start
                raw_ecg = np.asarray(ecg_data["samples"])[e0:e1]
                filtered_ecg = apply_filter(raw_ecg, header.fs, config.filter_config)
                acceleration = None
                if acc_data is not None:
                    acc_fs = float(acc_data["fs"])
                    a0 = int(round((start - block_start) * acc_fs / header.fs))
                    a1 = int(round((end - block_start) * acc_fs / header.fs))
                    acceleration = np.asarray(acc_data["samples"])[a0:a1]
                features = extract_feature_map(
                    filtered_ecg,
                    acceleration,
                    record.metadata,
                    config.feature_config,
                    acceleration_fs=float(acc_data["fs"]) if acc_data is not None else 1.0,
                )
                row: dict[str, Any] = {
                    "record_id": record.record_id,
                    "subject_id": record.subject_id,
                    "i0": start,
                    "i1": end,
                    "label": label,
                    **features,
                }
                rows.append(row)
            if progress:
                fraction = (record_index + (block_index + 1) / total_blocks) / len(record_ids)
                progress(fraction, f"Preparing {record_id}: block {block_index + 1}/{total_blocks}")
    if not rows:
        raise RuntimeError("No labelled windows were produced")
    frame = pd.DataFrame(rows)
    if config.balance == "undersample":
        counts = frame["label"].value_counts()
        target = int(counts.min())
        frame = (
            frame.groupby("label", group_keys=False)
            .apply(lambda group: group.sample(target, random_state=config.random_seed), include_groups=True)
            .sample(frac=1.0, random_state=config.random_seed)
            .reset_index(drop=True)
        )
    frame.attrs["build_diagnostics"] = {
        "eligible_windows": int(eligible_windows),
        "produced_windows_before_class_balance": int(len(rows)),
        "max_windows_per_record": int(config.max_windows_per_record),
        "capped_records": capped_records,
        "processing_seconds": float(
            __import__("time").perf_counter() - build_started
        ),
    }
    if progress:
        progress(1.0, f"Prepared {len(frame):,} windows")
    return frame


def _evaluation_splits(
    frame: pd.DataFrame, config: TrainingConfig
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    from sklearn.model_selection import StratifiedGroupKFold, train_test_split

    labels = frame["label"].to_numpy(dtype=int)
    selection_scope, _ = _selection_definition(config)
    if selection_scope == "class" and int(config.target_class) not in set(
        int(value) for value in np.unique(labels)
    ):
        raise ValueError(
            f"Target class {config.target_class} is absent from the prepared dataset"
        )
    indices = np.arange(len(frame))
    validation_fraction = float(config.validation_fraction)
    final_fraction = float(config.final_test_fraction)
    if not 0.05 <= validation_fraction <= 0.45:
        raise ValueError("Validation fraction must be between 0.05 and 0.45")
    if not 0.05 <= final_fraction <= 0.45:
        raise ValueError("Final-test fraction must be between 0.05 and 0.45")
    if validation_fraction + final_fraction >= 0.8:
        raise ValueError("Validation and final-test fractions leave too little training data")
    groups = frame["subject_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if config.split_by_subject and len(unique_groups) < 3:
        raise ValueError(
            "A person-disjoint train/validation/final-test evaluation requires at least three subjects"
        )
    if not config.split_by_subject:
        train_validation, final_test = train_test_split(
            indices,
            test_size=final_fraction,
            random_state=config.random_seed,
            stratify=labels,
        )
        relative_validation = validation_fraction / (1.0 - final_fraction)
        train, validation = train_test_split(
            train_validation,
            test_size=relative_validation,
            random_state=config.random_seed + 1,
            stratify=labels[train_validation],
        )
        return [(np.sort(train), np.sort(validation), np.sort(final_test))]
    strategy = str(config.split_strategy or "stratified_holdout").lower()
    if strategy == "stratified_group_kfold":
        folds = max(3, min(int(config.group_folds), len(unique_groups)))
        splitter = StratifiedGroupKFold(
            n_splits=folds, shuffle=True, random_state=config.random_seed
        )
        held_out = [
            np.asarray(test, dtype=int)
            for _, test in splitter.split(indices, labels, groups)
        ]
        output: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for final_index, final_test in enumerate(held_out):
            for validation_index, validation in enumerate(held_out):
                if validation_index == final_index:
                    continue
                excluded = np.concatenate((final_test, validation))
                train = np.setdiff1d(indices, excluded, assume_unique=False)
                if _split_has_classes(labels, train, validation, final_test):
                    output.append((train, validation, final_test))
        if not output:
            raise RuntimeError(
                "No Stratified Group K-Fold combination contains every class in all three partitions"
            )
        return output
    if strategy != "stratified_holdout":
        raise ValueError(
            "Split strategy must be stratified_holdout or stratified_group_kfold"
        )
    final_folds = max(2, min(len(unique_groups), round(1.0 / final_fraction)))
    outer = StratifiedGroupKFold(
        n_splits=final_folds, shuffle=True, random_state=config.random_seed
    )
    validation_relative = validation_fraction / (1.0 - final_fraction)
    for train_validation, final_test in outer.split(indices, labels, groups):
        remaining_groups = groups[train_validation]
        validation_folds = max(
            2, min(len(np.unique(remaining_groups)), round(1.0 / validation_relative))
        )
        inner = StratifiedGroupKFold(
            n_splits=validation_folds,
            shuffle=True,
            random_state=config.random_seed + 1,
        )
        for train_relative, validation_relative_indices in inner.split(
            train_validation,
            labels[train_validation],
            remaining_groups,
        ):
            train = train_validation[train_relative]
            validation = train_validation[validation_relative_indices]
            if _split_has_classes(labels, train, validation, final_test):
                return [(np.sort(train), np.sort(validation), np.sort(final_test))]
    raise RuntimeError(
        "No stratified person-disjoint holdout contains every quality class in train, validation, and final test"
    )


def _split_has_classes(
    labels: np.ndarray, *partitions: np.ndarray
) -> bool:
    required = set(int(value) for value in np.unique(labels))
    return all(set(int(value) for value in np.unique(labels[part])) == required for part in partitions)


def _predict_probabilities(estimator: Any, matrix: np.ndarray) -> np.ndarray | None:
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(matrix), dtype=float)
    return None


def _selection_score(
    truth: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray | None,
    config: TrainingConfig,
) -> float:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
    )

    scope, metric = _selection_definition(config)
    target = int(config.target_class)
    if metric == "loss":
        if probabilities is None:
            score = -1e9
        elif scope == "class":
            classes = sorted(int(value) for value in np.unique(truth))
            if target not in classes or probabilities.shape[1] != len(classes):
                score = -1e9
            else:
                target_probability = probabilities[:, classes.index(target)]
                binary_probabilities = np.column_stack(
                    (1.0 - target_probability, target_probability)
                )
                score = -float(
                    log_loss((truth == target).astype(int), binary_probabilities, labels=[0, 1])
                )
        else:
            classes = sorted(int(value) for value in np.unique(truth))
            score = -float(log_loss(truth, probabilities, labels=classes))
    elif metric == "accuracy" and scope == "class":
        actual_target = truth == target
        score = float(
            np.sum(actual_target & (predicted == target))
            / max(1, int(np.sum(actual_target)))
        )
    elif metric == "accuracy":
        score = float(accuracy_score(truth, predicted))
    elif metric == "balanced_accuracy" and scope == "class":
        score = float(
            balanced_accuracy_score(truth == target, predicted == target)
        )
    elif metric == "balanced_accuracy":
        score = float(balanced_accuracy_score(truth, predicted))
    elif metric == "precision" and scope == "class":
        score = float(
            precision_score(
                truth, predicted, labels=[target], average="macro", zero_division=0
            )
        )
    elif metric == "precision":
        score = float(
            precision_score(truth, predicted, average="macro", zero_division=0)
        )
    elif metric == "recall" and scope == "class":
        score = float(
            recall_score(
                truth, predicted, labels=[target], average="macro", zero_division=0
            )
        )
    elif metric == "recall":
        score = float(
            recall_score(truth, predicted, average="macro", zero_division=0)
        )
    elif metric == "f1" and scope == "class":
        score = float(
            f1_score(
                truth, predicted, labels=[target], average="macro", zero_division=0
            )
        )
    elif metric == "f1":
        score = float(f1_score(truth, predicted, average="macro", zero_division=0))
    else:
        raise ValueError(f"Unsupported selection metric: {config.selection_metric}")
    if config.error_penalty_enabled:
        dangerous_actual = truth == int(config.dangerous_actual_class)
        dangerous = float(
            np.sum(
                dangerous_actual
                & (predicted == int(config.dangerous_predicted_class))
            )
            / max(1, int(np.sum(dangerous_actual)))
        )
        score -= float(config.dangerous_error_penalty) * float(dangerous)
    return score


def _selection_definition(config: TrainingConfig) -> tuple[str, str]:
    """Normalize current and legacy selection settings to scope + metric."""
    metric = str(config.selection_metric or "accuracy").strip().lower()
    scope = str(getattr(config, "selection_scope", "overall") or "overall").strip().lower()
    if metric.startswith("class_"):
        scope, metric = "class", metric.removeprefix("class_")
    elif metric == "macro_f1":
        scope, metric = "overall", "f1"
    if scope not in {"overall", "class"}:
        raise ValueError(f"Unsupported selection scope: {scope}")
    if metric not in {"loss", "accuracy", "balanced_accuracy", "precision", "recall", "f1"}:
        raise ValueError(f"Unsupported selection metric: {config.selection_metric}")
    return scope, metric


def _selection_label(config: TrainingConfig) -> str:
    scope, metric = _selection_definition(config)
    name = metric.replace("_", " ").title()
    return f"Overall {name}" if scope == "overall" else f"Class {config.target_class} {name}"


def _subject_allocation(
    frame: pd.DataFrame,
    train: np.ndarray,
    validation: np.ndarray,
    final_test: np.ndarray,
) -> dict[str, Any]:
    subjects = frame["subject_id"].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    for subject in sorted(np.unique(subjects)):
        subject_total = int(np.sum(subjects == subject))
        values: dict[str, Any] = {"subject_id": subject, "total": subject_total}
        for name, partition in (
            ("train", train),
            ("validation", validation),
            ("final_test", final_test),
        ):
            count = int(np.sum(subjects[partition] == subject))
            values[f"{name}_count"] = count
            values[f"{name}_percent"] = 100.0 * count / max(1, subject_total)
        rows.append(values)
    return {
        "train_subjects": sorted(np.unique(subjects[train]).tolist()),
        "validation_subjects": sorted(np.unique(subjects[validation]).tolist()),
        "final_test_subjects": sorted(np.unique(subjects[final_test]).tolist()),
        "subject_allocation": rows,
    }


def _evaluation_warnings(
    validation_counts: dict[int, int],
    final_counts: dict[int, int],
    minimum: int,
    classes: list[int],
) -> list[str]:
    warnings: list[str] = []
    for partition, counts in (("Validation", validation_counts), ("Final test", final_counts)):
        for quality_class in classes:
            count = int(counts.get(quality_class, 0))
            if count < minimum:
                warnings.append(
                    f"{partition} Class {quality_class} contains only {count:,} samples. "
                    "Per-class metrics may be statistically unstable."
                )
    return warnings


def _wilson_interval(successes: int, total: int) -> list[float] | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * np.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [max(0.0, float(center - margin)), min(1.0, float(center + margin))]


def _confidence_intervals(
    truth: np.ndarray,
    predicted: np.ndarray,
    classes: list[int],
    seed: int,
) -> dict[str, Any]:
    from sklearn.metrics import precision_recall_fscore_support

    output: dict[str, Any] = {
        "accuracy": _wilson_interval(int(np.sum(truth == predicted)), len(truth)),
        "per_class": {},
    }
    for value in classes:
        tp = int(np.sum((truth == value) & (predicted == value)))
        predicted_positive = int(np.sum(predicted == value))
        actual_positive = int(np.sum(truth == value))
        output["per_class"][str(value)] = {
            "precision": _wilson_interval(tp, predicted_positive),
            "recall": _wilson_interval(tp, actual_positive),
        }
    if len(truth) >= 20:
        generator = np.random.default_rng(seed)
        macro_precision_values: list[float] = []
        macro_recall_values: list[float] = []
        macro_f1_values: list[float] = []
        class_values: dict[int, list[float]] = {value: [] for value in classes}
        for _ in range(200):
            sample = generator.integers(0, len(truth), size=len(truth))
            sampled_truth, sampled_prediction = truth[sample], predicted[sample]
            class_precision, class_recall, class_scores, _ = precision_recall_fscore_support(
                sampled_truth,
                sampled_prediction,
                labels=classes,
                zero_division=0,
            )
            macro_precision_values.append(float(np.mean(class_precision)))
            macro_recall_values.append(float(np.mean(class_recall)))
            macro_f1_values.append(float(np.mean(class_scores)))
            for index, value in enumerate(classes):
                class_values[value].append(float(class_scores[index]))
        output["balanced_accuracy"] = [
            float(value) for value in np.percentile(macro_recall_values, [2.5, 97.5])
        ]
        output["precision_macro"] = [
            float(value) for value in np.percentile(macro_precision_values, [2.5, 97.5])
        ]
        output["recall_macro"] = [
            float(value) for value in np.percentile(macro_recall_values, [2.5, 97.5])
        ]
        output["macro_f1"] = [
            float(value) for value in np.percentile(macro_f1_values, [2.5, 97.5])
        ]
        for value in classes:
            output["per_class"][str(value)]["f1"] = [
                float(item) for item in np.percentile(class_values[value], [2.5, 97.5])
            ]
    return output


def _frame_processing_seconds(frame: pd.DataFrame) -> float | None:
    diagnostics = frame.attrs.get("build_diagnostics", {})
    value = diagnostics.get("processing_seconds")
    return float(value) if value is not None else None


def train_model(
    frame: pd.DataFrame,
    config: TrainingConfig,
    progress: ProgressCallback | None = None,
    cancel: threading.Event | None = None,
) -> TrainingResult:
    """Train with a leakage-safe train/validation/final-test evaluation design."""
    started = __import__("time").perf_counter()
    splits = _evaluation_splits(frame, config)
    results: list[TrainingResult] = []
    for index, split in enumerate(splits):
        _check_cancel(cancel)
        candidate_progress = None
        if progress:
            candidate_progress = lambda value, text, i=index: progress(
                (i + value) / len(splits),
                f"Evaluation split {i + 1}/{len(splits)}: {text}",
            )
        result = _train_model_single(
            frame,
            config,
            progress=candidate_progress,
            cancel=cancel,
            split_indices=split,
        )
        results.append(result)
    selected_index = int(
        np.argmax(
            [float(result.metrics["validation_objective_score"]) for result in results]
        )
    )
    selected = results[selected_index]
    selected.metrics["evaluation_candidate_count"] = len(results)
    selected.metrics["selected_evaluation_candidate"] = selected_index + 1
    selected.metrics["evaluation_candidates"] = [
        {
            "candidate": index + 1,
            "selected": index == selected_index,
            "validation_objective_score": result.metrics["validation_objective_score"],
            "final_accuracy": result.metrics["accuracy"],
            "final_balanced_accuracy": result.metrics["balanced_accuracy"],
            "final_macro_f1": result.metrics["f1_macro"],
            "dangerous_error_count": result.metrics["dangerous_error_count"],
            "train_subjects": result.metrics["train_subjects"],
            "validation_subjects": result.metrics["validation_subjects"],
            "final_test_subjects": result.metrics["final_test_subjects"],
        }
        for index, result in enumerate(results)
    ]
    selected.metrics["training_seconds"] = float(
        __import__("time").perf_counter() - started
    )
    selected.metrics["data_preparation_seconds"] = _frame_processing_seconds(frame)
    if progress:
        progress(
            1.0,
            f"Finished {len(results)} evaluation split(s); selected {selected_index + 1}",
        )
    return selected


def _train_model_single(
    frame: pd.DataFrame,
    config: TrainingConfig,
    progress: ProgressCallback | None = None,
    cancel: threading.Event | None = None,
    split_indices: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> TrainingResult:
    if frame is None or frame.empty or "label" not in frame:
        raise ValueError("A non-empty prepared dataset is required")
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    feature_columns = [column for column in frame.columns if column not in META_COLUMNS]
    if not feature_columns:
        raise ValueError("Prepared dataset has no feature columns")
    matrix = frame[feature_columns].to_numpy(dtype=float)
    labels = frame["label"].to_numpy(dtype=int)
    medians = np.nanmedian(matrix, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    invalid = ~np.isfinite(matrix)
    if invalid.any():
        matrix[invalid] = np.take(medians, np.where(invalid)[1])
    if split_indices is None:
        split_indices = _evaluation_splits(frame, config)[0]
    train_indices, validation_indices, final_test_indices = split_indices
    x_train, x_validation = matrix[train_indices], matrix[validation_indices]
    y_train, y_validation = labels[train_indices], labels[validation_indices]
    x_final, y_final = matrix[final_test_indices], labels[final_test_indices]
    if len(np.unique(y_train)) < 2:
        raise RuntimeError("The training split contains fewer than two quality classes")
    equalization = str(config.class_equalization or "none").strip().lower()
    if equalization not in {"none", "oversample", "undersample"}:
        raise ValueError(
            "Class equalization must be none, oversample, or undersample"
        )
    train_counts_before = _class_counts(y_train)
    validation_counts_before = _class_counts(y_validation)
    final_counts = _class_counts(y_final)
    all_classes = set(int(value) for value in np.unique(labels))
    if config.equalize_validation and set(validation_counts_before) != all_classes:
        raise RuntimeError(
            "The validation split does not contain every quality class; "
            "change the split seed/fraction or disable person-disjoint splitting"
        )
    if equalization != "none":
        x_train, y_train = _resample_classes(
            x_train, y_train, equalization, config.random_seed
        )
        if config.equalize_validation:
            x_validation, y_validation = _resample_classes(
                x_validation, y_validation, equalization, config.random_seed + 1
            )
    train_counts_after = _class_counts(y_train)
    validation_counts_after = _class_counts(y_validation)
    # Resampling and class weights should not be combined: doing both would
    # count the minority correction twice and distort the fitted model.
    class_weight = (
        "balanced"
        if config.class_weight_balanced and equalization == "none"
        else None
    )
    penalty_weights = np.ones(len(y_train), dtype=float)
    if config.error_penalty_enabled:
        penalty_weights[y_train == int(config.dangerous_actual_class)] *= max(
            1.0, float(config.dangerous_error_penalty)
        )
    loss_curve: list[float] = []
    accuracy_curve: list[float] = []
    best_epoch = 0
    early_stopped = False
    training_backend = "scikit-learn CPU"
    model_type = config.model_type.strip()
    if progress:
        progress(0.02, f"Training {model_type}")
    _check_cancel(cancel)
    if model_type == "MLP":
        hidden = tuple(int(value.strip()) for value in config.mlp_hidden_layers.split(",") if value.strip())
        if not hidden or any(value < 1 for value in hidden):
            raise ValueError("MLP hidden layers must be positive comma-separated integers")
        if config.mlp_epochs < 1:
            raise ValueError("MLP epochs must be positive")
        if config.early_stopping_patience < 1:
            raise ValueError("Early-stopping patience must be positive")
        if config.early_stopping_min_delta < 0:
            raise ValueError("Early-stopping minimum improvement cannot be negative")
        scaler = StandardScaler() if config.standardize else None
        x_train_fit = scaler.fit_transform(x_train) if scaler else x_train
        x_validation_fit = scaler.transform(x_validation) if scaler else x_validation
        backend = str(config.mlp_backend or "sklearn").strip().lower()
        if backend not in {"sklearn", "auto", "pytorch_cuda"}:
            raise ValueError("MLP backend must be sklearn, auto, or pytorch_cuda")
        use_cuda = backend in {"auto", "pytorch_cuda"} and cuda_device_name() is not None
        if backend == "pytorch_cuda" and not use_cuda:
            raise RuntimeError(
                "PyTorch CUDA was requested, but no compatible CUDA device is available"
            )
        if use_cuda:
            (
                base,
                loss_curve,
                accuracy_curve,
                best_epoch,
                early_stopped,
                training_backend,
            ) = _train_torch_mlp(
                x_train_fit,
                y_train,
                x_validation_fit,
                y_validation,
                hidden,
                config,
                balanced_weights=class_weight is not None,
                progress=progress,
                cancel=cancel,
            )
        else:
            base = MLPClassifier(
                hidden_layer_sizes=hidden,
                learning_rate_init=config.learning_rate,
                max_iter=1,
                warm_start=True,
                random_state=config.random_seed,
                batch_size="auto",
            )
            import copy
            import warnings
            from sklearn.exceptions import ConvergenceWarning

            best_accuracy = float("-inf")
            best_parameters: tuple[list[np.ndarray], list[np.ndarray]] | None = None
            epochs_without_improvement = 0
            for epoch in range(config.mlp_epochs):
                _check_cancel(cancel)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    mlp_weights = penalty_weights.copy()
                    if class_weight is not None:
                        from sklearn.utils.class_weight import compute_sample_weight

                        mlp_weights *= compute_sample_weight("balanced", y_train)
                    base.fit(x_train_fit, y_train, sample_weight=mlp_weights)
                loss_curve.append(float(getattr(base, "loss_", np.nan)))
                validation_objective = float(
                    _selection_score(
                        y_validation,
                        base.predict(x_validation_fit),
                        _predict_probabilities(base, x_validation_fit),
                        config,
                    )
                )
                accuracy_curve.append(
                    -validation_objective
                    if _selection_definition(config)[1] == "loss"
                    else validation_objective
                )
                if validation_objective > best_accuracy + config.early_stopping_min_delta:
                    best_accuracy = validation_objective
                    best_epoch = epoch + 1
                    best_parameters = (
                        copy.deepcopy(base.coefs_),
                        copy.deepcopy(base.intercepts_),
                    )
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                if progress:
                    progress((epoch + 1) / config.mlp_epochs, f"MLP epoch {epoch + 1}/{config.mlp_epochs}")
                if (
                    config.early_stopping
                    and epochs_without_improvement >= config.early_stopping_patience
                ):
                    early_stopped = True
                    if progress:
                        progress(
                            (epoch + 1) / config.mlp_epochs,
                            f"Early stopping at epoch {epoch + 1}; restoring epoch {best_epoch}",
                        )
                    break
            if config.early_stopping and best_parameters is not None:
                base.coefs_, base.intercepts_ = best_parameters
        estimator = Pipeline([("scaler", scaler), ("model", base)]) if scaler else base
    elif model_type == "Logistic Regression":
        base = LogisticRegression(
            max_iter=3000,
            class_weight=class_weight,
            random_state=config.random_seed,
        )
        estimator = Pipeline([("scaler", StandardScaler()), ("model", base)]) if config.standardize else base
        if config.standardize:
            estimator.fit(x_train, y_train, model__sample_weight=penalty_weights)
        else:
            estimator.fit(x_train, y_train, sample_weight=penalty_weights)
    elif model_type == "Random Forest":
        estimator = RandomForestClassifier(
            n_estimators=config.rf_estimators,
            max_depth=config.rf_max_depth or None,
            class_weight=class_weight,
            n_jobs=config.parallel_jobs,
            random_state=config.random_seed,
        )
        estimator.fit(x_train, y_train, sample_weight=penalty_weights)
    elif model_type == "Gradient Boosting":
        estimator = GradientBoostingClassifier(random_state=config.random_seed)
        if class_weight is not None:
            from sklearn.utils.class_weight import compute_sample_weight

            estimator.fit(
                x_train,
                y_train,
                sample_weight=(
                    compute_sample_weight("balanced", y_train) * penalty_weights
                ),
            )
        else:
            estimator.fit(x_train, y_train, sample_weight=penalty_weights)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    _check_cancel(cancel)
    validation_predictions = estimator.predict(x_validation)
    validation_probabilities = _predict_probabilities(estimator, x_validation)
    validation_objective = _selection_score(
        y_validation, validation_predictions, validation_probabilities, config
    )
    predictions = estimator.predict(x_final)
    classes = sorted(int(value) for value in np.unique(labels))
    matrix_confusion = confusion_matrix(y_final, predictions, labels=classes)
    report = classification_report(
        y_final, predictions, labels=classes, output_dict=True, zero_division=0
    )
    allocation = _subject_allocation(
        frame, train_indices, validation_indices, final_test_indices
    )
    warnings = _evaluation_warnings(
        validation_counts_before,
        final_counts,
        int(config.minimum_evaluation_samples),
        sorted(all_classes),
    )
    confidence_intervals = _confidence_intervals(
        y_final, predictions, classes, config.random_seed
    )
    dangerous_errors = int(
        np.sum(
            (y_final == int(config.dangerous_actual_class))
            & (predictions == int(config.dangerous_predicted_class))
        )
    )
    metrics = {
        "accuracy": float(accuracy_score(y_final, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_final, predictions)),
        "precision_macro": float(
            precision_score(y_final, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_final, predictions, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_final, predictions, average="macro")),
        "f1_weighted": float(
            f1_score(y_final, predictions, average="weighted", zero_division=0)
        ),
        "classes": classes,
        "train_windows": int(len(y_train)),
        "train_windows_original": int(len(train_indices)),
        "validation_windows": int(len(y_validation)),
        "validation_windows_original": int(len(validation_indices)),
        "test_windows": int(len(y_final)),
        "test_windows_original": int(len(final_test_indices)),
        "test_windows_used": int(len(y_final)),
        "final_test_windows": int(len(y_final)),
        "final_test_windows_original": int(len(final_test_indices)),
        "class_equalization": equalization,
        "validation_equalized": bool(
            config.equalize_validation and equalization != "none"
        ),
        "epochs_requested": int(config.mlp_epochs if model_type == "MLP" else 1),
        "epochs_completed": int(len(loss_curve) if model_type == "MLP" else 1),
        "best_epoch": int(best_epoch if model_type == "MLP" else 1),
        "early_stopping_enabled": bool(
            model_type == "MLP" and config.early_stopping
        ),
        "early_stopped": bool(early_stopped),
        "early_stopping_patience": int(config.early_stopping_patience),
        "early_stopping_min_delta": float(config.early_stopping_min_delta),
        "training_backend": training_backend,
        "selection_metric": _selection_definition(config)[1],
        "selection_scope": _selection_definition(config)[0],
        "selection_target": _selection_label(config),
        "target_class": (
            int(config.target_class)
            if _selection_definition(config)[0] == "class"
            else None
        ),
        "validation_objective_score": float(validation_objective),
        "dangerous_error_penalty_enabled": bool(config.error_penalty_enabled),
        "dangerous_actual_class": int(config.dangerous_actual_class),
        "dangerous_predicted_class": int(config.dangerous_predicted_class),
        "dangerous_error_penalty": float(config.dangerous_error_penalty),
        "dangerous_error_count": dangerous_errors,
        "dangerous_error_rate": float(
            dangerous_errors
            / max(1, int(np.sum(y_final == int(config.dangerous_actual_class))))
        ),
        "train_class_counts_before": train_counts_before,
        "train_class_counts_after": train_counts_after,
        "validation_class_counts_before": validation_counts_before,
        "validation_class_counts_after": validation_counts_after,
        "test_class_counts_before": final_counts,
        "test_class_counts_after": final_counts,
        "test_class_counts": final_counts,
        "train_subjects": allocation["train_subjects"],
        "validation_subjects": allocation["validation_subjects"],
        "final_test_subjects": allocation["final_test_subjects"],
        "subject_allocation": allocation["subject_allocation"],
        "split_by_subject": bool(config.split_by_subject),
        "split_strategy": config.split_strategy,
        "evaluation_warnings": warnings,
        "confidence_intervals_95": confidence_intervals,
        "classification_report": report,
    }
    if progress:
        progress(1.0, f"Finished: accuracy={metrics['accuracy']:.3f}")
    return TrainingResult(
        estimator=estimator,
        feature_columns=feature_columns,
        metrics=metrics,
        confusion_matrix=matrix_confusion,
        classes=classes,
        loss_curve=loss_curve,
        accuracy_curve=accuracy_curve,
        config=config,
    )


def _train_torch_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    hidden: tuple[int, ...],
    config: TrainingConfig,
    balanced_weights: bool,
    progress: ProgressCallback | None,
    cancel: threading.Event | None,
) -> tuple[
    TorchMLPClassifier,
    list[float],
    list[float],
    int,
    bool,
    str,
]:
    import copy
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda:0")
    device_name = str(torch.cuda.get_device_name(device))
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    classes = np.asarray(sorted(int(value) for value in np.unique(y_train)), dtype=int)
    class_to_index = {value: index for index, value in enumerate(classes.tolist())}
    y_train_indices = np.asarray(
        [class_to_index[int(value)] for value in y_train], dtype=np.int64
    )
    y_validation_indices = np.asarray(
        [class_to_index[int(value)] for value in y_validation], dtype=np.int64
    )
    dimensions = [int(x_train.shape[1]), *hidden, len(classes)]
    layers: list[nn.Module] = []
    for index in range(len(dimensions) - 1):
        layers.append(nn.Linear(dimensions[index], dimensions[index + 1]))
        if index < len(dimensions) - 2:
            layers.append(nn.ReLU())
    model = nn.Sequential(*layers).to(device)
    class_weights = None
    if balanced_weights:
        counts = np.bincount(y_train_indices, minlength=len(classes)).astype(float)
        values = len(y_train_indices) / np.maximum(1.0, len(classes) * counts)
        class_weights = torch.as_tensor(values, dtype=torch.float32, device=device)
    if config.error_penalty_enabled:
        if class_weights is None:
            class_weights = torch.ones(len(classes), dtype=torch.float32, device=device)
        matches = np.where(classes == int(config.dangerous_actual_class))[0]
        if len(matches):
            class_weights[int(matches[0])] *= max(
                1.0, float(config.dangerous_error_penalty)
            )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_tensor = torch.as_tensor(np.asarray(x_train, dtype=np.float32))
    label_tensor = torch.as_tensor(y_train_indices, dtype=torch.long)
    # Large batches keep modern GPUs occupied while retaining a bounded VRAM
    # footprint for very large prepared datasets.
    batch_size = min(16384, max(512, len(train_tensor) // 8 or 512))
    generator = torch.Generator().manual_seed(config.random_seed)
    loader = DataLoader(
        TensorDataset(train_tensor, label_tensor),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=True,
        num_workers=0,
    )
    validation_tensor = torch.as_tensor(
        np.asarray(x_validation, dtype=np.float32)
    )
    loss_curve: list[float] = []
    accuracy_curve: list[float] = []
    best_accuracy = float("-inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    epochs_without_improvement = 0
    early_stopped = False
    for epoch in range(config.mlp_epochs):
        _check_cancel(cancel)
        model.train()
        total_loss = 0.0
        total_samples = 0
        for batch_values, batch_labels in loader:
            batch_values = batch_values.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_values)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item()) * len(batch_values)
            total_samples += len(batch_values)
        loss_curve.append(total_loss / max(1, total_samples))
        model.eval()
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(validation_tensor), 32768):
                batch = validation_tensor[start : start + 32768].to(
                    device, non_blocking=True
                )
                predictions.append(
                    torch.argmax(model(batch), dim=1).cpu().numpy()
                )
        predicted_indices = (
            np.concatenate(predictions)
            if predictions
            else np.asarray([], dtype=int)
        )
        validation_accuracy = float(
            np.mean(predicted_indices == y_validation_indices)
        )
        accuracy_curve.append(validation_accuracy)
        if validation_accuracy > best_accuracy + config.early_stopping_min_delta:
            best_accuracy = validation_accuracy
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if progress:
            progress(
                (epoch + 1) / config.mlp_epochs,
                f"CUDA MLP epoch {epoch + 1}/{config.mlp_epochs} on {device_name}",
            )
        if (
            config.early_stopping
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            early_stopped = True
            break
    if config.early_stopping and best_state is not None:
        model.load_state_dict(copy.deepcopy(best_state))
    linear_layers = [layer for layer in model if isinstance(layer, nn.Linear)]
    estimator = TorchMLPClassifier(hidden, config.learning_rate, f"PyTorch CUDA ({device_name})")
    estimator.classes_ = classes
    estimator.n_features_in_ = int(x_train.shape[1])
    estimator.coefs_ = [
        layer.weight.detach().cpu().numpy().T.copy() for layer in linear_layers
    ]
    estimator.intercepts_ = [
        layer.bias.detach().cpu().numpy().copy() for layer in linear_layers
    ]
    del model, train_tensor, validation_tensor
    torch.cuda.empty_cache()
    return (
        estimator,
        loss_curve,
        accuracy_curve,
        best_epoch,
        early_stopped,
        f"PyTorch CUDA ({device_name})",
    )


def save_dataframe(frame: pd.DataFrame, path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() == ".parquet":
        frame.to_parquet(target, index=False)
    elif target.suffix.lower() == ".csv":
        frame.to_csv(target, index=False)
    else:
        raise ValueError("Prepared datasets must use .parquet or .csv")
    diagnostics = frame.attrs.get("build_diagnostics")
    if isinstance(diagnostics, dict):
        target.with_suffix(target.suffix + ".diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2), encoding="utf-8"
        )
    return target


def load_dataframe(path: str | Path) -> pd.DataFrame:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() == ".parquet":
        frame = pd.read_parquet(source)
    elif source.suffix.lower() == ".csv":
        frame = pd.read_csv(source)
    else:
        raise ValueError("Prepared datasets must use .parquet or .csv")
    diagnostics_path = source.with_suffix(source.suffix + ".diagnostics.json")
    if diagnostics_path.is_file():
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        if isinstance(diagnostics, dict):
            frame.attrs["build_diagnostics"] = diagnostics
    return frame


def dataset_summary(frame: pd.DataFrame) -> dict[str, Any]:
    counts = frame["label"].value_counts().sort_index().to_dict() if "label" in frame else {}
    diagnostics = frame.attrs.get("build_diagnostics", {})
    return {
        "windows": len(frame),
        "records": int(frame["record_id"].nunique()) if "record_id" in frame else 0,
        "subjects": int(frame["subject_id"].nunique()) if "subject_id" in frame else 0,
        "features": len([column for column in frame.columns if column not in META_COLUMNS]),
        "class_counts": {int(key): int(value) for key, value in counts.items()},
        "eligible_windows": int(diagnostics.get("eligible_windows", len(frame))),
        "max_windows_per_record": int(
            diagnostics.get("max_windows_per_record", 0)
        ),
        "capped_records": dict(diagnostics.get("capped_records", {})),
    }


def save_training_report(result: TrainingResult, path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    report = {
        "metrics": result.metrics,
        "classes": result.classes,
        "confusion_matrix": result.confusion_matrix.tolist(),
        "feature_columns": result.feature_columns,
        "loss_curve": result.loss_curve,
        "accuracy_curve": result.accuracy_curve,
        "training_config": asdict(result.config),
    }
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return target


def _base_intervals(groups: list[list[tuple[int, int, int]]], source: str):
    source = source.lower().strip()
    if source == "consensus":
        return groups[3]
    if source.startswith("annotator"):
        return groups[int(source[-1]) - 1]
    return groups[3] or groups[0]


def _group_windows(
    windows: list[tuple[int, int, int]], fs: float, window_seconds: float
) -> list[list[tuple[int, int, int]]]:
    groups: list[list[tuple[int, int, int]]] = []
    maximum_span = int(round(fs * 120.0))
    maximum_gap = int(round(fs * max(window_seconds, 1.0)))
    for window in windows:
        if not groups:
            groups.append([window])
            continue
        current = groups[-1]
        group_start = current[0][0]
        group_end = current[-1][1]
        if window[0] - group_end <= maximum_gap and window[1] - group_start <= maximum_span:
            current.append(window)
        else:
            groups.append([window])
    return groups


def _check_cancel(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise RuntimeError("Operation cancelled")


def _resample_classes(
    matrix: np.ndarray, labels: np.ndarray, strategy: str, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    random = np.random.default_rng(seed)
    classes, counts = np.unique(labels, return_counts=True)
    target = int(np.max(counts) if strategy == "oversample" else np.min(counts))
    selected: list[np.ndarray] = []
    for quality_class, count in zip(classes, counts):
        indices = np.flatnonzero(labels == quality_class)
        selected.append(
            random.choice(indices, size=target, replace=int(count) < target)
        )
    combined = np.concatenate(selected)
    random.shuffle(combined)
    return matrix[combined], labels[combined]


def _class_counts(labels: np.ndarray) -> dict[int, int]:
    classes, counts = np.unique(labels, return_counts=True)
    return {
        int(quality_class): int(count)
        for quality_class, count in zip(classes, counts)
    }
