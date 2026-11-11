from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ekg_monitor.infrastructure.butqdb_repository import ButqdbRepository
from ekg_monitor.infrastructure.model_repository import save_model_bundle
from ekg_monitor.infrastructure.training_report_exporter import (
    export_training_artifacts,
)

from .models import DatasetBuildConfig, FeatureConfig, FilterConfig, TrainingConfig
from .training_service import (
    META_COLUMNS,
    build_dataset,
    load_dataframe,
    save_dataframe,
    save_training_report,
    train_model,
)


ProgressCallback = Callable[[float, str], None]
FEATURE_GROUPS = ("ecg_time", "ecg_spectrum", "acceleration", "acceleration_diff", "demographics")
MODEL_TYPES = ("MLP", "Logistic Regression", "Random Forest", "Gradient Boosting")


@dataclass(slots=True)
class OptimizationConfig:
    record_ids: list[str]
    window_lengths: list[float]
    varied_features: list[str]
    model_types: list[str]
    mlp_architectures: list[str] = field(default_factory=lambda: ["128,64"])
    dataset_root: str = "butqdb"
    database_root: str = "DBs"
    model_root: str = "Models"
    run_name: str = ""
    overlap: float = 0.5
    label_source: str = "consensus"
    max_windows_per_record: int = 0
    spectrum_bins: int = 32
    validation_fraction: float = 0.15
    final_test_fraction: float = 0.15
    split_strategy: str = "stratified_holdout"
    group_folds: int = 5
    random_seed: int = 42
    split_by_subject: bool = True
    standardize: bool = True
    class_equalization: str = "oversample"
    equalize_validation: bool = True
    class_weight_balanced: bool = True
    mlp_epochs: int = 200
    learning_rate: float = 0.001
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.001
    selection_metric: str = "accuracy"
    selection_scope: str = "overall"
    target_class: int = 0
    error_penalty_enabled: bool = True
    dangerous_error_penalty: float = 5.0
    minimum_evaluation_samples: int = 100
    rf_estimators: int = 300
    rf_max_depth: int = 0
    max_parallel_workers: int = 0
    ram_fraction: float = 0.75

    def validate(self) -> None:
        if not self.record_ids:
            raise ValueError("Select at least one annotated record")
        if not self.window_lengths or any(value <= 0 for value in self.window_lengths):
            raise ValueError("Window lengths must be positive")
        if not self.varied_features:
            raise ValueError("Select at least one feature group to vary")
        unknown_features = set(self.varied_features) - set(FEATURE_GROUPS)
        if unknown_features:
            raise ValueError(f"Unknown feature groups: {sorted(unknown_features)}")
        if not self.model_types:
            raise ValueError("Select at least one model family")
        unknown_models = set(self.model_types) - set(MODEL_TYPES)
        if unknown_models:
            raise ValueError(f"Unknown model types: {sorted(unknown_models)}")
        if "MLP" in self.model_types and not self.mlp_architectures:
            raise ValueError("Enter at least one MLP hidden-layer architecture")
        if self.mlp_epochs < 1 or self.early_stopping_patience < 1:
            raise ValueError("Epochs and early-stopping patience must be positive")
        if self.early_stopping_min_delta < 0:
            raise ValueError("Minimum validation improvement cannot be negative")
        if not 0.05 <= self.validation_fraction <= 0.45:
            raise ValueError("Validation fraction must be between 0.05 and 0.45")
        if not 0.05 <= self.final_test_fraction <= 0.45:
            raise ValueError("Final-test fraction must be between 0.05 and 0.45")
        if self.validation_fraction + self.final_test_fraction >= 0.8:
            raise ValueError("Validation and final-test fractions leave too little training data")
        if self.group_folds < 3:
            raise ValueError("Stratified Group K-Fold requires at least three folds")
        class_scoped = self.selection_scope == "class" or self.selection_metric.startswith(
            "class_"
        )
        if class_scoped and self.target_class not in (1, 2, 3):
            raise ValueError("Target quality class must be 1, 2, or 3")
        if self.dangerous_error_penalty < 1.0:
            raise ValueError("Dangerous-error penalty must be at least 1")
        if self.minimum_evaluation_samples < 1:
            raise ValueError("Minimum evaluation samples must be positive")
        if self.max_windows_per_record < 0:
            raise ValueError("Maximum windows per record cannot be negative")
        if self.max_parallel_workers < 0:
            raise ValueError("Parallel workers cannot be negative")
        if not 0.25 <= self.ram_fraction <= 0.9:
            raise ValueError("RAM usage fraction must be between 0.25 and 0.90")
        if self.selection_metric not in {
            "loss", "accuracy", "balanced_accuracy", "precision", "recall", "f1",
            "macro_f1", "class_precision", "class_recall", "class_f1",
        }:
            raise ValueError("Unsupported model-selection target")
        if self.selection_scope not in {"overall", "class"}:
            raise ValueError("Optimization scope must be overall or class")
        if self.split_strategy not in {
            "stratified_holdout", "stratified_group_kfold"
        }:
            raise ValueError("Unsupported evaluation design")


@dataclass(slots=True)
class OptimizationRun:
    rows: list[dict[str, Any]]
    run_name: str
    database_folder: Path
    model_folder: Path
    results_csv: Path
    results_xlsx: Path
    resource_summary: str = ""


@dataclass(slots=True)
class _CandidateTask:
    candidate_name: str
    candidate_folder: str
    dataset_name: str
    dataset_path: str
    window_seconds: float
    features: tuple[str, ...]
    model_type: str
    architecture: str
    build_config: DatasetBuildConfig
    training_config: TrainingConfig
    inner_threads: int = 1


def parse_window_lengths(text: str) -> list[float]:
    values: list[float] = []
    for part in str(text).replace(";", ",").split(","):
        if not part.strip():
            continue
        value = float(part.strip().replace(",", "."))
        if value <= 0:
            raise ValueError("Window lengths must be positive")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("Enter comma-separated window lengths, for example 3,5,10")
    return values


def parse_mlp_architectures(text: str) -> list[str]:
    architectures: list[str] = []
    for part in re.split(r"[;\n]+", str(text)):
        part = part.strip()
        if not part:
            continue
        layers = [int(value.strip()) for value in part.split(",") if value.strip()]
        if not layers or any(value < 1 for value in layers):
            raise ValueError("MLP architectures must contain positive layer sizes")
        normalized = ",".join(str(value) for value in layers)
        if normalized not in architectures:
            architectures.append(normalized)
    if not architectures:
        raise ValueError(
            "Enter semicolon-separated MLP architectures, for example 128,64;64,128,64"
        )
    return architectures


def feature_combinations(selected: list[str]) -> list[tuple[str, ...]]:
    ordered = [name for name in FEATURE_GROUPS if name in selected]
    return [
        values
        for size in range(1, len(ordered) + 1)
        for values in combinations(ordered, size)
    ]


def optimization_size(config: OptimizationConfig) -> tuple[int, int]:
    datasets = len(config.window_lengths) * len(feature_combinations(config.varied_features))
    models_per_dataset = sum(
        len(config.mlp_architectures) if model == "MLP" else 1
        for model in config.model_types
    )
    return datasets, datasets * models_per_dataset


def run_optimization(
    repository: ButqdbRepository,
    config: OptimizationConfig,
    progress: ProgressCallback | None = None,
    cancel: threading.Event | None = None,
) -> OptimizationRun:
    config.validate()
    repository.scan(config.dataset_root)
    feature_sets = feature_combinations(config.varied_features)
    model_variants = _model_variants(config)
    dataset_count = len(config.window_lengths) * len(feature_sets)
    candidate_count = dataset_count * len(model_variants)
    # One superset extraction per window length replaces a complete signal
    # reread for every feature subset.
    preparation_count = len(config.window_lengths)
    total_units = max(1, preparation_count + candidate_count)
    completed_units = 0
    run_name = _unique_run_name(config)
    database_folder = Path(config.database_root).expanduser().resolve() / run_name
    model_folder = Path(config.model_root).expanduser().resolve() / run_name
    database_folder.mkdir(parents=True, exist_ok=False)
    model_folder.mkdir(parents=True, exist_ok=False)
    (model_folder / "search_config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    rows: list[dict[str, Any]] = []
    tasks: list[_CandidateTask] = []
    candidate_index = 0
    maximum_dataset_bytes = 0
    for window_seconds in config.window_lengths:
        _check_cancel(cancel)
        master_config = DatasetBuildConfig(
            window_seconds=window_seconds,
            overlap=config.overlap,
            label_source=config.label_source,
            max_windows_per_record=config.max_windows_per_record,
            balance="none",
            random_seed=config.random_seed,
            filter_config=FilterConfig(remove_dc=True, filter_kind="none"),
            feature_config=_feature_config(
                tuple(config.varied_features), config.spectrum_bins
            ),
        )
        try:
            master_frame = build_dataset(
                repository,
                config.record_ids,
                master_config,
                progress=(
                    lambda value, text, base=completed_units, window=window_seconds: progress(
                        (base + value) / total_units,
                        f"{window:g}s feature superset: {text}",
                    )
                    if progress
                    else None
                ),
                cancel=cancel,
            )
        except Exception as exc:
            for features in feature_sets:
                dataset_name = _dataset_name(window_seconds, features)
                for model_type, architecture in model_variants:
                    candidate_index += 1
                    rows.append(
                        _failure_row(
                            dataset_name,
                            window_seconds,
                            features,
                            model_type,
                            f"Dataset preparation failed: {exc}",
                            architecture,
                            _candidate_name(
                                candidate_index,
                                dataset_name,
                                model_type,
                                architecture,
                            ),
                        )
                    )
            skipped = len(feature_sets) * len(model_variants)
            completed_units += 1 + skipped
            if progress:
                progress(
                    completed_units / total_units,
                    f"Skipped all {window_seconds:g}s candidates: {exc}",
                )
            continue
        completed_units += 1
        for features in feature_sets:
            _check_cancel(cancel)
            dataset_name = _dataset_name(window_seconds, features)
            dataset_directory = database_folder / dataset_name
            dataset_directory.mkdir(parents=True, exist_ok=False)
            build_config = replace(
                master_config,
                feature_config=_feature_config(features, config.spectrum_bins),
            )
            try:
                frame = _select_feature_subset(master_frame, features)
                maximum_dataset_bytes = max(
                    maximum_dataset_bytes,
                    int(frame.memory_usage(index=True, deep=True).sum()),
                )
                dataset_path = save_dataframe(
                    frame, dataset_directory / "prepared_dataset.parquet"
                )
                dataset_path.with_suffix(dataset_path.suffix + ".config.json").write_text(
                    json.dumps(asdict(build_config), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception as exc:
                for model_type, architecture in model_variants:
                    candidate_index += 1
                    rows.append(
                        _failure_row(
                            dataset_name,
                            window_seconds,
                            features,
                            model_type,
                            f"Dataset derivation failed: {exc}",
                            architecture,
                            _candidate_name(
                                candidate_index,
                                dataset_name,
                                model_type,
                                architecture,
                            ),
                        )
                    )
                    completed_units += 1
                continue
            for model_type, architecture in model_variants:
                candidate_index += 1
                candidate_name = _candidate_name(
                    candidate_index, dataset_name, model_type, architecture
                )
                candidate_folder = model_folder / candidate_name
                candidate_folder.mkdir(parents=True, exist_ok=False)
                training_config = _training_config(
                    config,
                    model_type,
                    architecture,
                )
                tasks.append(
                    _CandidateTask(
                        candidate_name=candidate_name,
                        candidate_folder=str(candidate_folder),
                        dataset_name=dataset_name,
                        dataset_path=str(dataset_path),
                        window_seconds=window_seconds,
                        features=features,
                        model_type=model_type,
                        architecture=architecture,
                        build_config=build_config,
                        training_config=training_config,
                    )
                )
        del master_frame

    cpu_tasks = tasks
    cpu_workers, inner_threads, memory_available, worker_memory = _resource_plan(
        config, maximum_dataset_bytes, len(cpu_tasks), False
    )
    for task in cpu_tasks:
        task.inner_threads = inner_threads
        task.training_config = replace(
            task.training_config, parallel_jobs=inner_threads
        )
    resource_summary = (
        f"Preparation: {preparation_count} feature-superset build(s) instead of "
        f"{dataset_count} repeated builds. CPU: {cpu_workers} parallel process(es) "
        f"x {inner_threads} thread(s); RAM budget {config.ram_fraction:.0%} of "
        f"{_format_bytes(memory_available)} available (~{_format_bytes(worker_memory)}/worker). "
        "All candidates use the same scikit-learn algorithms as single-model training; CUDA is disabled for reproducible parity."
    )
    (model_folder / "resource_plan.json").write_text(
        json.dumps(
            {
                "summary": resource_summary,
                "cpu_workers": cpu_workers,
                "threads_per_worker": inner_threads,
                "available_memory_bytes": memory_available,
                "estimated_worker_memory_bytes": worker_memory,
                "gpu": None,
                "gpu_task_count": 0,
                "cpu_task_count": len(cpu_tasks),
                "superset_builds": preparation_count,
                "derived_datasets": dataset_count,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if progress:
        progress(completed_units / total_units, resource_summary)
    completed_candidates = len(rows)
    future_tasks: dict[Any, _CandidateTask] = {}
    with ProcessPoolExecutor(max_workers=max(1, cpu_workers)) as cpu_executor:
        for task in cpu_tasks:
            future_tasks[cpu_executor.submit(_train_candidate_task, task)] = task
        for future in as_completed(future_tasks):
            task = future_tasks[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                rows.append(
                    _failure_row(
                        task.dataset_name,
                        task.window_seconds,
                        task.features,
                        task.model_type,
                        str(exc),
                        task.architecture,
                        task.candidate_name,
                    )
                )
            completed_units += 1
            completed_candidates += 1
            if progress:
                progress(
                    min(1.0, completed_units / total_units),
                    f"Completed {completed_candidates}/{candidate_count} model candidates",
                )
            if cancel is not None and cancel.is_set():
                for pending in future_tasks:
                    pending.cancel()
                raise RuntimeError("Operation cancelled")
    rows.sort(key=lambda row: str(row.get("candidate", "")))
    _mark_best_models(rows)
    results = pd.DataFrame(rows)
    results_csv = model_folder / "search_results.csv"
    results_xlsx = model_folder / "search_results.xlsx"
    results.to_csv(results_csv, index=False)
    results.to_excel(results_xlsx, index=False, sheet_name="Optimization Results")
    if progress:
        progress(1.0, f"Optimization complete: {len(rows)} result rows")
    return OptimizationRun(
        rows=rows,
        run_name=run_name,
        database_folder=database_folder,
        model_folder=model_folder,
        results_csv=results_csv,
        results_xlsx=results_xlsx,
        resource_summary=resource_summary,
    )


def _select_feature_subset(
    frame: pd.DataFrame, features: tuple[str, ...]
) -> pd.DataFrame:
    selected = set(features)
    columns: list[str] = []
    for column in frame.columns:
        if column in META_COLUMNS:
            columns.append(column)
            continue
        group = _feature_group_for_column(column)
        if group in selected:
            columns.append(column)
    feature_columns = [column for column in columns if column not in META_COLUMNS]
    if not feature_columns:
        raise ValueError(f"No columns were produced for feature set {features}")
    subset = frame.loc[:, columns].copy()
    subset.attrs.update(frame.attrs)
    return subset


def _feature_group_for_column(column: str) -> str:
    if column.startswith("ecg_time_"):
        return "ecg_time"
    if column.startswith("ecg_fft_"):
        return "ecg_spectrum"
    if column.startswith("acc_norm_"):
        return "acceleration"
    if column.startswith("acc_diff_"):
        return "acceleration_diff"
    if column.startswith("demo_"):
        return "demographics"
    return ""


def _train_candidate_task(task: _CandidateTask) -> dict[str, Any]:
    candidate_folder = Path(task.candidate_folder)
    started = time.perf_counter()
    try:
        from threadpoolctl import threadpool_limits

        frame = load_dataframe(task.dataset_path)
        with threadpool_limits(limits=max(1, task.inner_threads)):
            result = train_model(frame, task.training_config)
        training_seconds = time.perf_counter() - started
        model_path = save_model_bundle(
            candidate_folder / "model_bundle.joblib",
            result.estimator,
            result.feature_columns,
            task.build_config.filter_config,
            task.build_config.feature_config,
            result.metrics,
            {
                "training": asdict(task.training_config),
                "dataset": asdict(task.build_config),
                "candidate": task.candidate_name,
            },
        )
        save_training_report(result, candidate_folder / "training_results.json")
        export_training_artifacts(
            result, task.build_config, frame, candidate_folder
        )
        return _result_row(
            task.candidate_name,
            task.dataset_name,
            Path(task.dataset_path),
            model_path,
            task.window_seconds,
            task.features,
            task.model_type,
            task.architecture,
            result.metrics,
            training_seconds,
        )
    except Exception as exc:
        candidate_folder.mkdir(parents=True, exist_ok=True)
        (candidate_folder / "error.txt").write_text(str(exc), encoding="utf-8")
        return _failure_row(
            task.dataset_name,
            task.window_seconds,
            task.features,
            task.model_type,
            str(exc),
            task.architecture,
            task.candidate_name,
        )


def _resource_plan(
    config: OptimizationConfig,
    maximum_dataset_bytes: int,
    cpu_task_count: int,
    has_gpu_tasks: bool,
) -> tuple[int, int, int, int]:
    logical_cpus = max(1, os.cpu_count() or 1)
    available_memory = max(1, _available_memory_bytes())
    worker_memory = max(384 * 1024**2, int(maximum_dataset_bytes * 6.0))
    if cpu_task_count <= 0:
        return 0, 1, available_memory, worker_memory
    reserved_cpus = 2 if has_gpu_tasks and logical_cpus > 4 else 0
    cpu_capacity = max(1, logical_cpus - reserved_cpus)
    requested = config.max_parallel_workers or cpu_capacity
    requested = max(1, min(requested, cpu_capacity))
    ram_budget = int(available_memory * config.ram_fraction)
    if has_gpu_tasks:
        ram_budget = max(worker_memory, ram_budget - worker_memory)
    workers_by_ram = max(1, ram_budget // worker_memory)
    workers = max(
        1,
        min(cpu_task_count, requested, cpu_capacity, int(workers_by_ram)),
    )
    threads_per_worker = max(1, cpu_capacity // workers)
    return workers, threads_per_worker, available_memory, worker_memory


def _available_memory_bytes() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", wintypes.DWORD),
                    ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
        except Exception:
            pass
    return 8 * 1024**3


def _format_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024.0 or unit == "TiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{amount:.1f} TiB"


def _model_variants(config: OptimizationConfig) -> list[tuple[str, str]]:
    variants: list[tuple[str, str]] = []
    for model_type in MODEL_TYPES:
        if model_type not in config.model_types:
            continue
        if model_type == "MLP":
            variants.extend((model_type, value) for value in config.mlp_architectures)
        else:
            variants.append((model_type, "-") )
    return variants


def _feature_config(features: tuple[str, ...], spectrum_bins: int) -> FeatureConfig:
    selected = set(features)
    return FeatureConfig(
        use_ecg_time="ecg_time" in selected,
        use_ecg_spectrum="ecg_spectrum" in selected,
        use_acceleration="acceleration" in selected,
        use_acceleration_diff="acceleration_diff" in selected,
        use_demographics="demographics" in selected,
        spectrum_bins=spectrum_bins,
    )


def _training_config(
    config: OptimizationConfig,
    model_type: str,
    architecture: str,
) -> TrainingConfig:
    return TrainingConfig(
        model_type=model_type,
        validation_fraction=config.validation_fraction,
        final_test_fraction=config.final_test_fraction,
        split_strategy=config.split_strategy,
        group_folds=config.group_folds,
        random_seed=config.random_seed,
        standardize=config.standardize,
        split_by_subject=config.split_by_subject,
        class_weight_balanced=config.class_weight_balanced,
        class_equalization=config.class_equalization,
        equalize_validation=config.equalize_validation,
        mlp_hidden_layers=architecture if model_type == "MLP" else "128,64",
        mlp_epochs=config.mlp_epochs,
        learning_rate=config.learning_rate,
        early_stopping=True,
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_min_delta=config.early_stopping_min_delta,
        selection_metric=config.selection_metric,
        selection_scope=config.selection_scope,
        target_class=config.target_class,
        error_penalty_enabled=config.error_penalty_enabled,
        dangerous_actual_class=3,
        dangerous_predicted_class=1,
        dangerous_error_penalty=config.dangerous_error_penalty,
        minimum_evaluation_samples=config.minimum_evaluation_samples,
        mlp_backend="sklearn",
        parallel_jobs=1,
        rf_estimators=config.rf_estimators,
        rf_max_depth=config.rf_max_depth,
    )


def _result_row(
    candidate_name: str,
    dataset_name: str,
    dataset_path: Path,
    model_path: Path,
    window_seconds: float,
    features: tuple[str, ...],
    model_type: str,
    architecture: str,
    metrics: dict[str, Any],
    training_seconds: float,
) -> dict[str, Any]:
    return {
        "candidate": candidate_name,
        "status": "ok",
        "best_for": "",
        "window_seconds": float(window_seconds),
        "features": "+".join(features),
        "model": model_type,
        "architecture": architecture,
        "accuracy": float(metrics.get("accuracy", math.nan)),
        "balanced_accuracy": float(metrics.get("balanced_accuracy", math.nan)),
        "macro_f1": float(metrics.get("f1_macro", math.nan)),
        "weighted_f1": float(metrics.get("f1_weighted", math.nan)),
        "train_windows": int(metrics.get("train_windows", 0)),
        "validation_windows": int(metrics.get("validation_windows", 0)),
        "final_test_windows": int(
            metrics.get("final_test_windows", metrics.get("test_windows", 0))
        ),
        "selection_metric": str(metrics.get("selection_metric", "accuracy")),
        "selection_scope": str(metrics.get("selection_scope", "overall")),
        "selection_target": str(metrics.get("selection_target", "Overall Accuracy")),
        "validation_objective_score": float(
            metrics.get("validation_objective_score", math.nan)
        ),
        "dangerous_error_count": int(metrics.get("dangerous_error_count", 0)),
        "evaluation_warnings": " | ".join(metrics.get("evaluation_warnings", [])),
        "best_epoch": int(metrics.get("best_epoch", 1)),
        "epochs_completed": int(metrics.get("epochs_completed", 1)),
        "early_stopped": bool(metrics.get("early_stopped", False)),
        "training_backend": str(metrics.get("training_backend", "")),
        "training_seconds": float(training_seconds),
        "artifact_bytes": int(model_path.stat().st_size),
        "dataset": dataset_name,
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "error": "",
    }


def _failure_row(
    dataset_name: str,
    window_seconds: float,
    features: tuple[str, ...],
    model_type: str,
    error: str,
    architecture: str = "-",
    candidate_name: str = "-",
) -> dict[str, Any]:
    return {
        "candidate": candidate_name,
        "status": "failed",
        "best_for": "",
        "window_seconds": float(window_seconds),
        "features": "+".join(features),
        "model": model_type,
        "architecture": architecture,
        "accuracy": math.nan,
        "balanced_accuracy": math.nan,
        "macro_f1": math.nan,
        "weighted_f1": math.nan,
        "train_windows": 0,
        "validation_windows": 0,
        "final_test_windows": 0,
        "selection_metric": "",
        "selection_scope": "",
        "selection_target": "",
        "validation_objective_score": math.nan,
        "dangerous_error_count": 0,
        "evaluation_warnings": "",
        "best_epoch": 0,
        "epochs_completed": 0,
        "early_stopped": False,
        "training_seconds": math.nan,
        "artifact_bytes": 0,
        "dataset": dataset_name,
        "dataset_path": "",
        "model_path": "",
        "error": error,
    }


def _mark_best_models(rows: list[dict[str, Any]]) -> None:
    successful = [row for row in rows if row.get("status") == "ok"]
    if not successful:
        return
    criteria = (
        ("accuracy", max, "accuracy"),
        ("balanced_accuracy", max, "balanced accuracy"),
        ("macro_f1", max, "macro F1"),
        ("training_seconds", min, "fastest"),
        ("artifact_bytes", min, "smallest"),
    )
    for key, selector, label in criteria:
        values = [float(row[key]) for row in successful if math.isfinite(float(row[key]))]
        if not values:
            continue
        best = selector(values)
        for row in successful:
            value = float(row[key])
            if math.isfinite(value) and math.isclose(value, best, rel_tol=1e-9, abs_tol=1e-12):
                current = [part for part in str(row["best_for"]).split(", ") if part]
                current.append(label)
                row["best_for"] = ", ".join(current)


def _dataset_name(window_seconds: float, features: tuple[str, ...]) -> str:
    window = f"{window_seconds:g}".replace(".", "p")
    abbreviations = {
        "ecg_time": "time",
        "ecg_spectrum": "fft",
        "acceleration": "acc",
        "acceleration_diff": "accdiff",
        "demographics": "demo",
    }
    return f"w{window}s_" + "-".join(abbreviations[value] for value in features)


def _candidate_name(
    index: int, dataset_name: str, model_type: str, architecture: str
) -> str:
    model = {
        "MLP": "mlp",
        "Logistic Regression": "logreg",
        "Random Forest": "rf",
        "Gradient Boosting": "gb",
    }[model_type]
    suffix = "_" + architecture.replace(",", "-") if model_type == "MLP" else ""
    return f"{index:04d}_{dataset_name}_{model}{suffix}"


def _unique_run_name(config: OptimizationConfig) -> str:
    requested = re.sub(r"[^A-Za-z0-9_.-]+", "_", config.run_name.strip()).strip("._")
    base = requested or datetime.now().strftime("optimization_%Y%m%d_%H%M%S")
    database_root = Path(config.database_root).expanduser().resolve()
    model_root = Path(config.model_root).expanduser().resolve()
    candidate = base
    index = 2
    while (database_root / candidate).exists() or (model_root / candidate).exists():
        candidate = f"{base}_{index}"
        index += 1
    return candidate


def _check_cancel(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise RuntimeError("Operation cancelled")
