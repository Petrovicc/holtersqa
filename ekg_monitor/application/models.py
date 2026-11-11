from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


FILTER_KINDS = (
    "none",
    "butter_bandpass",
    "butter_highpass",
    "butter_lowpass",
    "cheby1_bandpass",
    "cheby1_highpass",
    "cheby1_lowpass",
    "fir_bandpass",
    "fir_highpass",
    "fir_lowpass",
    "notch_50",
    "notch_60",
)


@dataclass(slots=True)
class FilterConfig:
    remove_dc: bool = True
    filter_kind: str = "butter_bandpass"
    order: int = 4
    low_hz: float = 0.5
    high_hz: float = 40.0
    ripple_db: float = 0.5

    def validate(self, fs: float | None = None) -> None:
        if self.filter_kind not in FILTER_KINDS:
            raise ValueError(f"Unsupported filter: {self.filter_kind}")
        if self.order < 1:
            raise ValueError("Filter order/taps must be positive")
        if self.low_hz < 0 or self.high_hz <= 0:
            raise ValueError("Filter frequencies must be positive")
        if "bandpass" in self.filter_kind and self.low_hz >= self.high_hz:
            raise ValueError("Band-pass low frequency must be below high frequency")
        if fs is not None and self.filter_kind != "none":
            if self.filter_kind == "notch_60":
                cutoff = 60.0
            elif self.filter_kind == "notch_50":
                cutoff = 50.0
            else:
                cutoff = self.high_hz
            if "highpass" in self.filter_kind:
                cutoff = self.low_hz
            if cutoff >= fs / 2:
                raise ValueError(f"Cutoff must be below Nyquist ({fs / 2:g} Hz)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FeatureConfig:
    use_ecg_time: bool = True
    use_ecg_spectrum: bool = True
    use_acceleration: bool = True
    use_acceleration_diff: bool = True
    use_demographics: bool = True
    spectrum_bins: int = 32

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DatasetBuildConfig:
    window_seconds: float = 10.0
    overlap: float = 0.5
    label_source: str = "consensus"
    # Zero means unlimited. A positive value intentionally samples each record
    # and can hide the expected difference between short and long windows.
    max_windows_per_record: int = 0
    balance: str = "none"
    random_seed: int = 42
    filter_config: FilterConfig = field(default_factory=FilterConfig)
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)


@dataclass(slots=True)
class TrainingConfig:
    model_type: str = "MLP"
    validation_fraction: float = 0.15
    final_test_fraction: float = 0.15
    split_strategy: str = "stratified_holdout"
    group_folds: int = 5
    random_seed: int = 42
    standardize: bool = True
    split_by_subject: bool = True
    class_weight_balanced: bool = True
    class_equalization: str = "oversample"
    equalize_validation: bool = True
    mlp_hidden_layers: str = "128,64"
    mlp_epochs: int = 50
    learning_rate: float = 0.001
    early_stopping: bool = True
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.001
    selection_metric: str = "accuracy"
    selection_scope: str = "overall"
    target_class: int = 0
    error_penalty_enabled: bool = False
    dangerous_actual_class: int = 3
    dangerous_predicted_class: int = 1
    dangerous_error_penalty: float = 5.0
    minimum_evaluation_samples: int = 100
    mlp_backend: str = "sklearn"
    parallel_jobs: int = -1
    rf_estimators: int = 300
    rf_max_depth: int = 0


@dataclass(slots=True)
class ClientStreamConfig:
    dataset_root: str
    subject_id: str
    host: str = "127.0.0.1"
    port: int = 50070
    interval_seconds: float = 2.0
    auto_reconnect: bool = True
    reconnect_limit: int = 0

    def validate(self) -> None:
        if not self.dataset_root:
            raise ValueError("Dataset folder is required")
        if not self.subject_id:
            raise ValueError("Subject ID is required")
        if not self.host:
            raise ValueError("Server IP/host is required")
        if not (1 <= self.port <= 65535):
            raise ValueError("Port must be in 1..65535")
        if not (1.0 <= self.interval_seconds <= 5.0):
            raise ValueError("Streaming interval must be between 1 and 5 seconds")


@dataclass(slots=True)
class SignalAssessment:
    quality_class: int
    quality_label: str
    quality_score: float
    reason: str
    bpm: float | None = None
    health_warning: str | None = None
    source: str = "heuristic"
    pulse_latest_rr_ms: float | None = None
    pulse_interval_count: int = 0
    pulse_window_seconds: float = 0.0


@dataclass(slots=True)
class FeedbackMessage:
    timestamp: str
    category: str
    severity: str
    message: str
