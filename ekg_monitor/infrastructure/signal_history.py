from __future__ import annotations

import re
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class HistoryBlock:
    samples: np.ndarray
    fs: float
    channel_names: list[str]
    units: list[str]
    start_seconds: float
    end_seconds: float


@dataclass(slots=True)
class _StreamState:
    path: Path
    fs: float
    channels: int
    channel_names: list[str]
    units: list[str]
    rows: int = 0


class DiskSignalHistory:
    """Append-only, disk-backed signal history for an active monitoring session."""

    def __init__(self, subject_id: str) -> None:
        safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", subject_id)[:64] or "subject"
        self._temporary = tempfile.TemporaryDirectory(prefix=f"ecg-{safe_id}-")
        self.root = Path(self._temporary.name)
        self._streams: dict[str, _StreamState] = {}
        self._lock = threading.RLock()
        self._closed = False

    def append(
        self,
        name: str,
        samples: np.ndarray,
        fs: float,
        channel_names: list[str],
        units: list[str],
    ) -> None:
        array = np.asarray(samples, dtype="<f4")
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if not len(array):
            return
        with self._lock:
            if self._closed:
                return
            state = self._streams.get(name)
            if state is None:
                state = _StreamState(
                    path=self.root / f"{name}.f32",
                    fs=float(fs),
                    channels=int(array.shape[1]),
                    channel_names=list(channel_names),
                    units=list(units),
                )
                self._streams[name] = state
            if state.channels != array.shape[1] or abs(state.fs - float(fs)) > 1e-6:
                raise ValueError(f"The {name} stream format changed during the session")
            with state.path.open("ab") as output:
                array.tofile(output)
            state.rows += int(array.shape[0])

    def duration(self, name: str = "ecg") -> float:
        with self._lock:
            state = self._streams.get(name)
            return state.rows / state.fs if state and state.fs > 0 else 0.0

    def read(self, name: str, start_seconds: float, end_seconds: float) -> HistoryBlock | None:
        with self._lock:
            state = self._streams.get(name)
            if state is None or state.rows <= 0 or self._closed:
                return None
            start_row = max(0, min(state.rows, int(round(start_seconds * state.fs))))
            end_row = max(start_row, min(state.rows, int(round(end_seconds * state.fs))))
            if end_row <= start_row:
                return None
            mapped = np.memmap(
                state.path,
                dtype="<f4",
                mode="r",
                shape=(state.rows, state.channels),
            )
            try:
                samples = np.asarray(mapped[start_row:end_row]).copy()
            finally:
                del mapped
            return HistoryBlock(
                samples=samples,
                fs=state.fs,
                channel_names=list(state.channel_names),
                units=list(state.units),
                start_seconds=start_row / state.fs,
                end_seconds=end_row / state.fs,
            )

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                name: {
                    "fs": state.fs,
                    "rows": state.rows,
                    "duration_seconds": state.rows / state.fs,
                    "channels": state.channels,
                    "units": list(state.units),
                }
                for name, state in self._streams.items()
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._streams.clear()
        self._temporary.cleanup()
