from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ekg_monitor.application.models import FeatureConfig, FilterConfig, SignalAssessment
from ekg_monitor.application.signal_analysis import features_for_columns


@dataclass(slots=True)
class LoadedModel:
    path: Path
    estimator: Any
    feature_columns: list[str]
    filter_config: FilterConfig = field(default_factory=FilterConfig)
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    window_seconds: float = 10.0
    preprocessing_complete: bool = False
    preprocessing_config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate_for_inference(self) -> None:
        if not self.preprocessing_complete:
            raise ValueError(
                "The model artifact has no complete preprocessing definition. "
                "Re-export it from the current ECG Model Training application."
            )
        if not self.feature_columns:
            raise ValueError("The model artifact has no input feature names")
        if self.window_seconds <= 0:
            raise ValueError("The model inference window must be positive")
        if self.feature_config.spectrum_bins < 1:
            raise ValueError("The model spectrum-bin count must be positive")
        self.filter_config.validate()
        expected_methods = {
            "input_signal": "filtered_ecg",
            "filter_method": "zero_phase_per_window",
            "feature_method": "extract_feature_map/v1",
        }
        for key, expected in expected_methods.items():
            actual = self.preprocessing_config.get(key)
            if self.preprocessing_config and actual != expected:
                raise ValueError(
                    f"Unsupported model preprocessing {key}={actual!r}; "
                    f"the server requires {expected!r}"
                )
        incompatible_columns = _incompatible_feature_columns(
            self.feature_columns, self.feature_config
        )
        if incompatible_columns:
            raise ValueError(
                "The model feature list conflicts with its feature configuration: "
                + ", ".join(incompatible_columns[:8])
            )

    def predict(
        self,
        ecg: np.ndarray,
        acceleration: np.ndarray | None,
        demographics: dict[str, Any],
        acceleration_fs: float = 1.0,
    ) -> SignalAssessment:
        if not self.feature_columns:
            raise ValueError("The model artifact does not describe its input features")
        vector = features_for_columns(
            self.feature_columns,
            ecg,
            acceleration,
            demographics,
            acceleration_fs=acceleration_fs,
            config=self.feature_config,
        )
        prediction = int(np.asarray(self.estimator.predict(vector)).reshape(-1)[0])
        score = 1.0
        if hasattr(self.estimator, "predict_proba"):
            probabilities = np.asarray(self.estimator.predict_proba(vector)).reshape(1, -1)[0]
            score = float(np.max(probabilities))
        labels = {1: "good", 2: "acceptable for QRS only", 3: "unusable"}
        return SignalAssessment(
            quality_class=prediction,
            quality_label=labels.get(prediction, f"class {prediction}"),
            quality_score=score,
            reason=f"Model prediction: class {prediction}",
            source="machine learning",
        )

    def describe(self) -> list[tuple[str, str]]:
        estimator = self.estimator
        final = estimator
        steps: list[str] = []
        if hasattr(estimator, "steps"):
            steps = [f"{name}: {type(value).__name__}" for name, value in estimator.steps]
            final = estimator.steps[-1][1]
        rows: list[tuple[str, str]] = [
            ("Artifact", str(self.path)),
            ("Type", type(final).__name__),
            ("Pipeline", " -> ".join(steps) if steps else "No pipeline"),
            ("Inputs", str(len(self.feature_columns))),
            ("Input signal", "ECG filtered with the exported training filter"),
            ("Inference window", f"{self.window_seconds:g} seconds"),
            (
                "Preprocessing",
                f"window={self.window_seconds:g}s; filter={self.filter_config.filter_kind}; "
                f"remove_dc={self.filter_config.remove_dc}; spectrum_bins={self.feature_config.spectrum_bins}",
            ),
            ("Input names", ", ".join(self.feature_columns) if self.feature_columns else "Unavailable"),
            (
                "Classes",
                str(
                    [
                        value.item() if isinstance(value, np.generic) else value
                        for value in list(getattr(final, "classes_", []))
                    ]
                ),
            ),
            ("Complexity", _complexity(final)),
        ]
        parameters = getattr(final, "get_params", lambda: {})()
        for key in sorted(parameters):
            value = parameters[key]
            text = str(value)
            rows.append((f"parameter.{key}", text[:500]))
        for key, value in sorted(self.metadata.items()):
            if key not in {"model", "feature_columns"}:
                rows.append((f"metadata.{key}", str(value)[:500]))
        return rows


def load_model(path: str | Path) -> LoadedModel:
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("joblib is required to load models") from exc
    artifact_path = Path(path).expanduser().resolve()
    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)
    loaded = joblib.load(artifact_path)
    metadata: dict[str, Any] = {}
    feature_config = FeatureConfig()
    window_seconds = 10.0
    preprocessing_complete = False
    if isinstance(loaded, dict) and "model" in loaded:
        estimator = loaded["model"]
        columns = [str(value) for value in loaded.get("feature_columns", [])]
        metadata = {key: value for key, value in loaded.items() if key != "model"}
        filter_values = loaded.get("filter_config")
        feature_values = loaded.get("feature_config")
        filter_config = FilterConfig(**(filter_values or {}))
        feature_config = FeatureConfig(**(feature_values or {}))
        preprocessing = loaded.get("preprocessing_config") or {}
        training_metadata = loaded.get("training_config") or {}
        dataset_metadata = (
            training_metadata.get("dataset", {})
            if isinstance(training_metadata, dict)
            else {}
        )
        window_value = preprocessing.get(
            "window_seconds", dataset_metadata.get("window_seconds")
        )
        if window_value is not None:
            window_seconds = float(window_value)
        methods_compatible = bool(
            (
                not preprocessing
                and loaded.get("artifact_format") == "ekg-monitor-model/2"
            )
            or all(
                preprocessing.get(key) == value
                for key, value in {
                    "input_signal": "filtered_ecg",
                    "filter_method": "zero_phase_per_window",
                    "feature_method": "extract_feature_map/v1",
                }.items()
            )
        )
        preprocessing_complete = bool(
            columns
            and isinstance(filter_values, dict)
            and isinstance(feature_values, dict)
            and window_value is not None
            and methods_compatible
        )
    else:
        estimator = loaded
        adjacent = artifact_path.with_name("results.json")
        if adjacent.is_file():
            metadata = json.loads(adjacent.read_text(encoding="utf-8"))
        columns = [str(value) for value in metadata.get("feature_columns", [])]
        filter_config = FilterConfig()
        preprocessing = {}
    if not hasattr(estimator, "predict"):
        raise TypeError("Loaded object is not a predictive model")
    return LoadedModel(
        path=artifact_path,
        estimator=estimator,
        feature_columns=columns,
        filter_config=filter_config,
        feature_config=feature_config,
        window_seconds=window_seconds,
        preprocessing_complete=preprocessing_complete,
        preprocessing_config=dict(preprocessing),
        metadata=metadata,
    )


def save_model_bundle(
    path: str | Path,
    estimator: Any,
    feature_columns: list[str],
    filter_config: FilterConfig,
    feature_config: FeatureConfig,
    metrics: dict[str, Any],
    training_config: dict[str, Any],
) -> Path:
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("joblib is required to save models") from exc
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "artifact_format": "ekg-monitor-model/2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": estimator,
        "feature_columns": list(feature_columns),
        "filter_config": asdict(filter_config),
        "feature_config": asdict(feature_config),
        "preprocessing_config": {
            "input_signal": "filtered_ecg",
            "filter_method": "zero_phase_per_window",
            "feature_method": "extract_feature_map/v1",
            "window_seconds": float(
                (
                    training_config.get("dataset", {})
                    if isinstance(training_config, dict)
                    else {}
                ).get("window_seconds", 10.0)
            ),
        },
        "metrics": _json_safe(metrics),
        "training_config": _json_safe(training_config),
    }
    joblib.dump(bundle, target)
    summary = {key: value for key, value in bundle.items() if key != "model"}
    target.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return target


def _complexity(estimator: Any) -> str:
    if hasattr(estimator, "hidden_layer_sizes"):
        layers = tuple(estimator.hidden_layer_sizes)
        weights = sum(np.asarray(value).size for value in getattr(estimator, "coefs_", []))
        biases = sum(np.asarray(value).size for value in getattr(estimator, "intercepts_", []))
        return f"hidden layers={layers}, trainable parameters={weights + biases:,}"
    if hasattr(estimator, "estimators_"):
        trees = np.asarray(estimator.estimators_, dtype=object).reshape(-1)
        nodes = sum(getattr(getattr(tree, "tree_", None), "node_count", 0) for tree in trees)
        return f"estimators={len(trees)}, total tree nodes={nodes:,}"
    coefficients = getattr(estimator, "coef_", None)
    if coefficients is not None:
        return f"coefficients={np.asarray(coefficients).size:,}"
    return "Not reported by estimator"


def _incompatible_feature_columns(
    columns: list[str], config: FeatureConfig
) -> list[str]:
    incompatible: list[str] = []
    for name in columns:
        if name.startswith("ecg_time_") and not config.use_ecg_time:
            incompatible.append(name)
        elif name.startswith("ecg_fft_"):
            match = re.search(r"_b(\d+)$", name)
            if not config.use_ecg_spectrum or (
                match is not None and int(match.group(1)) >= config.spectrum_bins
            ):
                incompatible.append(name)
        elif name.startswith("acc_norm_") and not config.use_acceleration:
            incompatible.append(name)
        elif name.startswith("acc_diff_") and not config.use_acceleration_diff:
            incompatible.append(name)
        elif name.startswith("demo_") and not config.use_demographics:
            incompatible.append(name)
    return incompatible


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value
