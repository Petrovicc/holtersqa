from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(slots=True)
class SignalHeader:
    record_name: str
    fs: float
    sample_count: int
    signal_count: int
    channel_names: list[str] = field(default_factory=list)
    units: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RecordInfo:
    record_id: str
    subject_id: str
    folder: Path
    ecg_base: Path | None
    acc_base: Path | None
    annotation_path: Path | None
    metadata: dict[str, Any]
    ecg_header: SignalHeader | None = None
    acc_header: SignalHeader | None = None

    def connection_metadata(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "health_registry_id": self.record_id,
            "record_id": self.record_id,
            "person_id": self.subject_id,
            **self.metadata,
        }
        if self.ecg_header:
            value["ecg"] = asdict(self.ecg_header)
        if self.acc_header:
            value["accelerometer"] = asdict(self.acc_header)
        return _json_safe(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if pd.isna(value):
        return None
    return value


def _parse_signal_line(line: str) -> tuple[str, str]:
    parts = line.split()
    channel_name = parts[-1] if parts else "signal"
    units = ""
    if len(parts) > 2 and "/" in parts[2]:
        units = parts[2].rsplit("/", 1)[-1]
    return channel_name, units


def read_wfdb_header(path: Path) -> SignalHeader:
    """Read the subset of a WFDB header needed by the UI without loading data."""
    with path.open("r", encoding="ascii", errors="replace") as stream:
        lines = [line.strip() for line in stream if line.strip() and not line.startswith("#")]
    if not lines:
        raise ValueError(f"Empty WFDB header: {path}")
    first = lines[0].split()
    if len(first) < 4:
        raise ValueError(f"Malformed WFDB header: {path}")
    name, signal_count, fs_text, sample_count = first[:4]
    fs = float(fs_text.split("/", 1)[0])
    channels: list[str] = []
    units: list[str] = []
    for line in lines[1 : 1 + int(signal_count)]:
        channel, unit = _parse_signal_line(line)
        channels.append(channel)
        units.append(unit)
    return SignalHeader(
        record_name=name,
        fs=fs,
        sample_count=int(sample_count),
        signal_count=int(signal_count),
        channel_names=channels,
        units=units,
    )


class ButqdbRepository:
    """Windowed access to the locally downloaded BUT QDB database."""

    def __init__(self, root: str | Path | None = None):
        self.root: Path | None = None
        self.records: dict[str, RecordInfo] = {}
        if root:
            self.scan(root)

    def scan(self, root: str | Path) -> list[RecordInfo]:
        candidate = Path(root).expanduser().resolve()
        if not candidate.is_dir():
            raise FileNotFoundError(candidate)
        metadata = self._read_subject_info(candidate / "subject-info.csv")
        records: dict[str, RecordInfo] = {}
        for folder in sorted((p for p in candidate.iterdir() if p.is_dir()), key=lambda p: p.name):
            record_id = folder.name
            ecg_base = folder / f"{record_id}_ECG"
            acc_base = folder / f"{record_id}_ACC"
            ann_path = folder / f"{record_id}_ANN.csv"
            ecg_base = ecg_base if _wfdb_pair_exists(ecg_base) else None
            acc_base = acc_base if _wfdb_pair_exists(acc_base) else None
            if ecg_base is None and acc_base is None:
                continue
            info = RecordInfo(
                record_id=record_id,
                subject_id=record_id[:3],
                folder=folder,
                ecg_base=ecg_base,
                acc_base=acc_base,
                annotation_path=ann_path if ann_path.is_file() else None,
                metadata=dict(metadata.get(record_id, metadata.get(record_id[:3], {}))),
            )
            if ecg_base:
                info.ecg_header = read_wfdb_header(ecg_base.with_suffix(".hea"))
            if acc_base:
                info.acc_header = read_wfdb_header(acc_base.with_suffix(".hea"))
            records[record_id] = info
        if not records:
            raise ValueError(f"No BUT QDB WFDB records found in {candidate}")
        self.root = candidate
        self.records = records
        return list(records.values())

    @staticmethod
    def _read_subject_info(path: Path) -> dict[str, dict[str, Any]]:
        if not path.is_file():
            return {}
        # BUT QDB uses semicolons; Sniffer keeps this robust for repackaged copies.
        sample = path.read_text(encoding="utf-8-sig", errors="replace")[:2048]
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=";,\t").delimiter
        except csv.Error:
            delimiter = ";"
        frame = pd.read_csv(path, sep=delimiter)
        frame.columns = [_normal_key(column) for column in frame.columns]
        id_column = next(
            (name for name in ("id", "record_id", "subject_id", "subject") if name in frame.columns),
            None,
        )
        if not id_column:
            return {}
        output: dict[str, dict[str, Any]] = {}
        for _, row in frame.iterrows():
            record_id = str(row[id_column]).strip()
            if record_id.endswith(".0"):
                record_id = record_id[:-2]
            values = {
                _normal_key(key): _json_safe(value)
                for key, value in row.to_dict().items()
                if key != id_column
            }
            output[record_id] = values
        return output

    def get(self, record_id: str) -> RecordInfo:
        try:
            return self.records[str(record_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown BUT QDB record: {record_id}") from exc

    def read_time_window(
        self, record_id: str, start_seconds: float, duration_seconds: float
    ) -> dict[str, dict[str, Any]]:
        info = self.get(record_id)
        if duration_seconds <= 0:
            raise ValueError("Duration must be positive")
        result: dict[str, dict[str, Any]] = {}
        for name, base, header in (
            ("ecg", info.ecg_base, info.ecg_header),
            ("acc", info.acc_base, info.acc_header),
        ):
            if base is None or header is None:
                continue
            start = max(0, int(round(start_seconds * header.fs)))
            end = min(header.sample_count, start + int(round(duration_seconds * header.fs)))
            if end <= start:
                continue
            result[name] = self._read_samples(base, start, end, header)
        return result

    def read_ecg_sample_window(
        self, record_id: str, start_sample: int, end_sample: int
    ) -> dict[str, dict[str, Any]]:
        info = self.get(record_id)
        if info.ecg_header is None or info.ecg_base is None:
            raise ValueError(f"Record {record_id} has no ECG")
        start_sample = max(0, int(start_sample))
        end_sample = min(info.ecg_header.sample_count, int(end_sample))
        if end_sample <= start_sample:
            raise ValueError("Empty ECG sample window")
        start_seconds = start_sample / info.ecg_header.fs
        duration_seconds = (end_sample - start_sample) / info.ecg_header.fs
        return self.read_time_window(record_id, start_seconds, duration_seconds)

    @staticmethod
    def _read_samples(
        base: Path, start: int, end: int, header: SignalHeader
    ) -> dict[str, Any]:
        try:
            import wfdb
        except ImportError as exc:
            raise RuntimeError("WFDB is required to read BUT QDB signals") from exc
        record = wfdb.rdrecord(str(base), sampfrom=start, sampto=end)
        signal = np.asarray(record.p_signal, dtype=np.float32)
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
        return {
            "samples": signal,
            "fs": float(record.fs),
            "channel_names": list(getattr(record, "sig_name", None) or header.channel_names),
            "units": list(getattr(record, "units", None) or header.units),
            "start_sample": start,
            "end_sample": end,
        }

    def annotation_groups(self, record_id: str) -> list[list[tuple[int, int, int]]]:
        """Return expert 1..3 and consensus intervals as zero-based half-open ranges."""
        info = self.get(record_id)
        if info.annotation_path is None:
            return [[], [], [], []]
        frame = pd.read_csv(info.annotation_path, header=None)
        values = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        groups: list[list[tuple[int, int, int]]] = []
        for offset in (0, 3, 6, 9):
            intervals: list[tuple[int, int, int]] = []
            for row in values:
                if len(row) <= offset + 2 or not np.all(np.isfinite(row[offset : offset + 3])):
                    continue
                start, end, label = (int(v) for v in row[offset : offset + 3])
                if start >= 1 and end >= start and label in (1, 2, 3):
                    intervals.append((start - 1, end, label))
            groups.append(intervals)
        return groups


def _wfdb_pair_exists(base: Path) -> bool:
    return base.with_suffix(".hea").is_file() and base.with_suffix(".dat").is_file()


def _normal_key(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def interval_label(
    groups: list[list[tuple[int, int, int]]], start: int, end: int, source: str
) -> int:
    source = source.lower().strip()
    if source == "consensus":
        return _dominant_interval_label(groups[3], start, end) if len(groups) > 3 else 0
    if source.startswith("annotator"):
        try:
            index = int(source[-1]) - 1
        except ValueError:
            return 0
        return _dominant_interval_label(groups[index], start, end) if 0 <= index < 3 else 0
    labels = [_dominant_interval_label(group, start, end) for group in groups[:3]]
    labels = [label for label in labels if label]
    if not labels:
        return 0
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=4)
    return int(np.argmax(counts[1:]) + 1)


def _dominant_interval_label(
    intervals: Iterable[tuple[int, int, int]], start: int, end: int
) -> int:
    overlap = [0, 0, 0, 0]
    for interval_start, interval_end, label in intervals:
        overlap[label] += max(0, min(end, interval_end) - max(start, interval_start))
    label = int(np.argmax(overlap[1:]) + 1)
    return label if overlap[label] >= max(1, (end - start) // 2) else 0
