from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

from .models import FeatureConfig, FilterConfig, SignalAssessment


STAT_NAMES = ("min", "max", "mean", "std", "rms", "ptp", "med", "mad")


def apply_filter(samples: np.ndarray, fs: float, config: FilterConfig) -> np.ndarray:
    values = np.asarray(samples, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("Signal must contain at least two samples")
    config.validate(fs)
    output = values.copy()
    if config.remove_dc:
        output -= np.mean(output, axis=0, keepdims=True)
    kind = config.filter_kind
    if kind == "none":
        return output
    try:
        from scipy import signal
    except ImportError as exc:
        raise RuntimeError("SciPy is required for filtering") from exc
    nyquist = fs / 2.0
    if kind.startswith("notch_"):
        frequency = 50.0 if kind == "notch_50" else 60.0
        b, a = signal.iirnotch(frequency, 30.0, fs=fs)
        return _safe_filtfilt(signal, b, a, output)
    family, filter_type = kind.split("_", 1)
    cutoff: float | list[float]
    if filter_type == "bandpass":
        cutoff = [config.low_hz, config.high_hz]
    elif filter_type == "highpass":
        cutoff = config.low_hz
    else:
        cutoff = config.high_hz
    if family == "butter":
        sos = signal.butter(config.order, cutoff, btype=filter_type, fs=fs, output="sos")
        return _safe_sosfiltfilt(signal, sos, output)
    if family == "cheby1":
        sos = signal.cheby1(
            config.order,
            max(0.01, config.ripple_db),
            cutoff,
            btype=filter_type,
            fs=fs,
            output="sos",
        )
        return _safe_sosfiltfilt(signal, sos, output)
    taps = max(5, config.order)
    if taps % 2 == 0:
        taps += 1
    if taps < 21:
        taps = 101
    pass_zero = filter_type == "lowpass"
    coefficients = signal.firwin(taps, cutoff, pass_zero=pass_zero, fs=fs)
    return _safe_filtfilt(signal, coefficients, [1.0], output)


def apply_filter_causal(
    samples: np.ndarray, fs: float, config: FilterConfig
) -> np.ndarray:
    """Filter a live/history prefix without changing earlier samples later.

    Unlike zero-phase ``filtfilt``, a causal filter never uses samples from the
    future. A feature already visible in retained history therefore remains
    unchanged when the next network packet arrives.
    """
    values = np.asarray(samples, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("Signal must contain at least two samples")
    config.validate(fs)
    output = values.copy()
    if not config.remove_dc and config.filter_kind == "none":
        return output
    try:
        from scipy import signal
    except ImportError as exc:
        raise RuntimeError("SciPy is required for filtering") from exc
    if config.remove_dc:
        # Stable streaming DC blocker with a 0.05 Hz corner. Subtracting the
        # mean of a moving plot window would rewrite previously shown values.
        pole = math.exp(-2.0 * math.pi * 0.05 / fs)
        output = signal.lfilter([1.0, -1.0], [1.0, -pole], output, axis=0)
    kind = config.filter_kind
    if kind == "none":
        return output
    if kind.startswith("notch_"):
        frequency = 50.0 if kind == "notch_50" else 60.0
        b, a = signal.iirnotch(frequency, 30.0, fs=fs)
        return signal.lfilter(b, a, output, axis=0)
    family, filter_type = kind.split("_", 1)
    if filter_type == "bandpass":
        cutoff: float | list[float] = [config.low_hz, config.high_hz]
    elif filter_type == "highpass":
        cutoff = config.low_hz
    else:
        cutoff = config.high_hz
    if family == "butter":
        sos = signal.butter(
            config.order, cutoff, btype=filter_type, fs=fs, output="sos"
        )
        return signal.sosfilt(sos, output, axis=0)
    if family == "cheby1":
        sos = signal.cheby1(
            config.order,
            max(0.01, config.ripple_db),
            cutoff,
            btype=filter_type,
            fs=fs,
            output="sos",
        )
        return signal.sosfilt(sos, output, axis=0)
    taps = max(5, config.order)
    if taps % 2 == 0:
        taps += 1
    if taps < 21:
        taps = 101
    coefficients = signal.firwin(
        taps, cutoff, pass_zero=filter_type == "lowpass", fs=fs
    )
    return signal.lfilter(coefficients, [1.0], output, axis=0)


def _safe_filtfilt(signal_module: Any, b: Any, a: Any, values: np.ndarray) -> np.ndarray:
    padlen = min(len(values) - 1, 3 * (max(len(a), len(b)) - 1))
    if padlen < 1:
        return values
    return signal_module.filtfilt(b, a, values, axis=0, padlen=padlen)


def _safe_sosfiltfilt(signal_module: Any, sos: Any, values: np.ndarray) -> np.ndarray:
    default_pad = 3 * (2 * len(sos) + 1)
    padlen = min(len(values) - 1, default_pad)
    if padlen < 1:
        return values
    return signal_module.sosfiltfilt(sos, values, axis=0, padlen=padlen)


def assess_signal(samples: np.ndarray, fs: float) -> SignalAssessment:
    signal = np.asarray(samples, dtype=float).reshape(-1)
    if fs <= 0 or len(signal) < max(64, int(fs * 1.5)):
        return SignalAssessment(3, "unusable", 0.0, "Window is too short")
    centered = signal - np.nanmean(signal)
    finite = np.isfinite(centered)
    if finite.mean() < 0.99:
        return SignalAssessment(3, "unusable", 0.0, "Signal contains invalid values")
    standard_deviation = float(np.std(centered))
    dynamic_range = float(np.ptp(centered))
    if standard_deviation < 1e-7 or dynamic_range < 1e-6:
        return SignalAssessment(3, "unusable", 0.0, "Flatline or missing electrodes")
    lower = float(np.min(centered) + dynamic_range * 0.01)
    upper = float(np.max(centered) - dynamic_range * 0.01)
    clipping_fraction = float(np.mean((centered <= lower) | (centered >= upper)))
    qrs_ratio = _band_energy_ratio(centered, fs)
    baseline_ratio = _frequency_energy_ratio(centered, fs, 0.0, 0.5)
    high_frequency_ratio = _frequency_energy_ratio(centered, fs, 40.0, min(fs / 2, 150.0))
    score = float(
        np.clip(
            0.55 * min(1.0, qrs_ratio / 0.18)
            + 0.25 * max(0.0, 1.0 - clipping_fraction * 8.0)
            + 0.20 * max(0.0, 1.0 - baseline_ratio - high_frequency_ratio),
            0.0,
            1.0,
        )
    )
    if clipping_fraction > 0.15 or qrs_ratio < 0.025:
        quality_class, label = 3, "unusable"
    elif score < 0.58 or qrs_ratio < 0.08:
        quality_class, label = 2, "acceptable for QRS only"
    else:
        quality_class, label = 1, "good"
    (
        bpm,
        pulse_reason,
        latest_rr_ms,
        interval_count,
        pulse_window_seconds,
    ) = estimate_pulse_details(signal, fs)
    health_warning = None
    if bpm is not None and bpm < 40:
        health_warning = f"Low pulse warning: {bpm:.1f} bpm"
    elif bpm is not None and bpm > 180:
        health_warning = f"High pulse warning: {bpm:.1f} bpm"
    reason = (
        f"QRS energy={qrs_ratio:.2f}, clipping={clipping_fraction:.2%}, "
        f"baseline/high-frequency noise={baseline_ratio + high_frequency_ratio:.2f}"
    )
    if bpm is None:
        reason += f"; pulse unavailable ({pulse_reason})"
    return SignalAssessment(
        quality_class=quality_class,
        quality_label=label,
        quality_score=score,
        reason=reason,
        bpm=bpm,
        health_warning=health_warning,
        pulse_latest_rr_ms=latest_rr_ms,
        pulse_interval_count=interval_count,
        pulse_window_seconds=pulse_window_seconds,
    )


def _frequency_energy_ratio(samples: np.ndarray, fs: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    window = np.hanning(len(samples))
    power = np.abs(np.fft.rfft(samples * window)) ** 2
    frequencies = np.fft.rfftfreq(len(samples), 1.0 / fs)
    total = float(np.sum(power)) + 1e-12
    return float(np.sum(power[(frequencies >= low) & (frequencies < high)]) / total)


def _band_energy_ratio(samples: np.ndarray, fs: float) -> float:
    return _frequency_energy_ratio(samples, fs, 5.0, min(20.0, fs / 2))


def estimate_pulse(samples: np.ndarray, fs: float) -> tuple[float | None, str]:
    bpm, reason, _latest_rr_ms, _interval_count, _window_seconds = estimate_pulse_details(
        samples, fs
    )
    return bpm, reason


def estimate_pulse_details(
    samples: np.ndarray, fs: float
) -> tuple[float | None, str, float | None, int, float]:
    values = np.asarray(samples, dtype=float).reshape(-1)
    window_seconds = len(values) / fs if fs > 0 else 0.0
    if fs <= 0 or len(values) < int(fs * 3):
        return None, "less than three seconds", None, 0, window_seconds
    values -= np.mean(values)
    width = max(1, int(0.02 * fs))
    smoothed = np.convolve(values, np.ones(width) / width, mode="same")
    derivative_energy = np.diff(smoothed, prepend=smoothed[0]) ** 2
    integration_width = max(1, int(0.08 * fs))
    integrated = np.convolve(
        derivative_energy, np.ones(integration_width) / integration_width, mode="same"
    )
    threshold = float(np.mean(integrated) + 1.4 * np.std(integrated))
    if not math.isfinite(threshold) or threshold <= 0:
        return None, "invalid detector threshold", None, 0, window_seconds
    refractory = max(1, int(0.25 * fs))
    peaks: list[int] = []
    cursor = 0
    while cursor < len(integrated):
        if integrated[cursor] <= threshold:
            cursor += 1
            continue
        stop = min(len(integrated), cursor + int(0.15 * fs))
        peak = cursor + int(np.argmax(integrated[cursor:stop]))
        peaks.append(peak)
        cursor = peak + refractory
    if len(peaks) < 2:
        return None, "not enough QRS candidates", None, 0, window_seconds
    rr = np.diff(np.asarray(peaks, dtype=float)) / fs
    rr = rr[(rr >= 0.25) & (rr <= 2.0)]
    if not len(rr):
        return None, "implausible RR intervals", None, 0, window_seconds
    recent_rr = rr[-5:]
    bpm = float(60.0 / np.median(recent_rr))
    return bpm, "ok", float(recent_rr[-1] * 1000.0), len(recent_rr), window_seconds


def extract_feature_map(
    ecg: np.ndarray | None,
    acceleration: np.ndarray | None,
    demographics: dict[str, Any] | None,
    config: FeatureConfig,
    acceleration_fs: float = 1.0,
) -> dict[str, float]:
    features: dict[str, float] = {}
    if ecg is not None:
        ecg_values = _two_dimensional(ecg)
        for channel in range(ecg_values.shape[1]):
            if config.use_ecg_time:
                _put_stats(features, f"ecg_time_c{channel}", ecg_values[:, channel])
            if config.use_ecg_spectrum:
                spectrum = _spectrum_bins(ecg_values[:, channel], config.spectrum_bins)
                for index, value in enumerate(spectrum):
                    features[f"ecg_fft_c{channel}_b{index}"] = value
    if acceleration is not None:
        acc_values = _two_dimensional(acceleration)
        if config.use_acceleration:
            for channel in range(acc_values.shape[1]):
                centered = acc_values[:, channel] - np.mean(acc_values[:, channel])
                _put_stats(features, f"acc_norm_c{channel}", centered)
        if config.use_acceleration_diff:
            differences = np.diff(acc_values, axis=0, prepend=acc_values[:1]) * max(
                1.0, float(acceleration_fs)
            )
            for channel in range(differences.shape[1]):
                _put_stats(features, f"acc_diff_c{channel}", differences[:, channel])
    if config.use_demographics:
        normalized = {_normal_key(k): v for k, v in (demographics or {}).items()}
        features.update(
            {
                "demo_gender": _gender_number(normalized.get("gender", normalized.get("sex"))),
                "demo_age": _number(normalized.get("age")),
                "demo_height": _number(normalized.get("height")),
                "demo_weight": _number(normalized.get("weight")),
                "demo_smoker": _boolean_number(
                    normalized.get("smoker", normalized.get("smoking_status"))
                ),
            }
        )
    return features


def features_for_columns(
    columns: list[str],
    ecg: np.ndarray,
    acceleration: np.ndarray | None,
    demographics: dict[str, Any],
    acceleration_fs: float = 1.0,
    config: FeatureConfig | None = None,
) -> np.ndarray:
    if config is None:
        highest_bin = max(
            (int(match.group(1)) for name in columns if (match := re.search(r"ecg_fft_c\d+_b(\d+)$", name))),
            default=31,
        )
        config = FeatureConfig(spectrum_bins=highest_bin + 1)
    feature_map = extract_feature_map(
        ecg, acceleration, demographics, config, acceleration_fs=acceleration_fs
    )
    return np.asarray([[feature_map.get(name, 0.0) for name in columns]], dtype=float)


def _two_dimensional(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array.reshape(-1, 1) if array.ndim == 1 else array


def _stats(values: np.ndarray) -> list[float]:
    array = np.asarray(values, dtype=float).reshape(-1)
    if not len(array):
        return [0.0] * len(STAT_NAMES)
    median = float(np.median(array))
    return [
        float(np.min(array)),
        float(np.max(array)),
        float(np.mean(array)),
        float(np.std(array)),
        float(np.sqrt(np.mean(array * array))),
        float(np.ptp(array)),
        median,
        float(np.median(np.abs(array - median))),
    ]


def _put_stats(target: dict[str, float], prefix: str, values: np.ndarray) -> None:
    for name, value in zip(STAT_NAMES, _stats(values)):
        target[f"{prefix}_{name}"] = value


def _spectrum_bins(values: np.ndarray, count: int) -> list[float]:
    array = np.asarray(values, dtype=float).reshape(-1)
    if not len(array):
        return [0.0] * count
    spectrum = np.abs(np.fft.rfft(array - np.mean(array)))[1:]
    if not len(spectrum):
        return [0.0] * count
    edges = np.linspace(0, len(spectrum), count + 1).astype(int)
    return [
        float(np.mean(spectrum[edges[index] : edges[index + 1]]))
        if edges[index + 1] > edges[index]
        else 0.0
        for index in range(count)
    ]


def _normal_key(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def _number(value: Any) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _gender_number(value: Any) -> float:
    text = str(value or "").strip().lower()
    if text in {"m", "male", "man", "1"}:
        return 1.0
    if text in {"f", "female", "woman", "0"}:
        return 0.0
    return _number(value)


def _boolean_number(value: Any) -> float:
    text = str(value or "").strip().lower()
    if text in {"true", "yes", "y", "1"}:
        return 1.0
    if text in {"false", "no", "n", "0"}:
        return 0.0
    return _number(value)
