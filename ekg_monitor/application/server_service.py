from __future__ import annotations

import json
import queue
import re
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ekg_monitor.infrastructure.model_repository import LoadedModel, load_model
from ekg_monitor.infrastructure.protocol import ProtocolError, decode_signal_blocks, recv_frame, send_frame
from ekg_monitor.infrastructure.signal_history import DiskSignalHistory

from .models import FilterConfig, SignalAssessment
from .signal_analysis import apply_filter, apply_filter_causal, assess_signal


@dataclass(slots=True)
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 50070
    max_clients: int = 16
    interval_override_seconds: float = 0.0
    save_chunks: bool = False
    save_root: str = "server_data_v2"

    def validate(self) -> None:
        if not self.host:
            raise ValueError("Host is required")
        if not 1 <= self.port <= 65535:
            raise ValueError("Port must be in 1..65535")
        if not 1 <= self.max_clients <= 1000:
            raise ValueError("Maximum clients must be in 1..1000")
        if self.interval_override_seconds and not 1 <= self.interval_override_seconds <= 5:
            raise ValueError("Server interval override must be 0 or between 1 and 5 seconds")


@dataclass(slots=True)
class ClientState:
    subject_id: str
    address: tuple[str, int]
    socket: socket.socket
    session_id: str
    metadata: dict[str, Any]
    interval_seconds: float
    history: DiskSignalHistory
    connected_at: float = field(default_factory=time.monotonic)
    last_packet_at: float = field(default_factory=time.monotonic)
    packet_count: int = 0
    bytes_received: int = 0
    ping_ms: float | None = None
    jitter_ms: float = 0.0
    processing_latency_ms: float | None = None
    last_interarrival_seconds: float | None = None
    last_signal_time: float = 0.0
    last_bpm: float | None = None
    last_assessment: SignalAssessment | None = None
    raw_ecg: np.ndarray | None = None
    filtered_ecg: np.ndarray | None = None
    acceleration: np.ndarray | None = None
    ecg_fs: float = 0.0
    acc_fs: float = 0.0
    filter_config: FilterConfig = field(default_factory=FilterConfig)
    acknowledged_sequences: dict[int, int] = field(default_factory=dict)
    feedback_times: dict[str, float] = field(default_factory=dict)
    send_lock: threading.Lock = field(default_factory=threading.Lock)
    lock: threading.RLock = field(default_factory=threading.RLock)
    connected: bool = True
    confusion_matrix: np.ndarray = field(
        default_factory=lambda: np.zeros((3, 3), dtype=np.int64)
    )
    labelled_predictions: int = 0


class ServerService:
    def __init__(self) -> None:
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.clients: dict[str, ClientState] = {}
        self.histories: dict[str, DiskSignalHistory] = {}
        self.clients_lock = threading.RLock()
        self.running = threading.Event()
        self._listener: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self.config = ServerConfig()
        self.model: LoadedModel | None = None
        self.model_lock = threading.RLock()

    def start(self, config: ServerConfig) -> None:
        config.validate()
        if self.running.is_set():
            raise RuntimeError("Server is already running")
        with self.model_lock:
            if self.model is None:
                raise RuntimeError("Import a trained model before starting the server")
        save_root = Path(config.save_root).expanduser().resolve()
        if config.save_chunks:
            save_root.mkdir(parents=True, exist_ok=True)
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((config.host, config.port))
        listener.listen(config.max_clients)
        listener.settimeout(1.0)
        self.config = config
        self._listener = listener
        self.running.set()
        self._accept_thread = threading.Thread(
            target=self._accept_loop, name="ecg-server-accept", daemon=True
        )
        self._accept_thread.start()
        self._emit("log", f"Listening on {config.host}:{config.port}")
        self._emit("server_status", "running")

    def stop(self) -> None:
        self.running.clear()
        listener, self._listener = self._listener, None
        if listener:
            try:
                listener.close()
            except OSError:
                pass
        with self.clients_lock:
            states = list(self.clients.values())
            self.clients.clear()
            histories = list(self.histories.values())
            self.histories.clear()
        for state in states:
            state.connected = False
            self._close_client_socket(state.socket)
        for history in {id(item): item for item in histories}.values():
            history.close()
        self._emit("clients_changed", None)
        self._emit("log", "Server stopped")
        self._emit("server_status", "stopped")

    def import_model(self, path: str | Path) -> LoadedModel:
        if self.running.is_set():
            raise RuntimeError("Stop the server before changing the loaded model")
        loaded = load_model(path)
        loaded.validate_for_inference()
        with self.model_lock:
            self.model = loaded
        with self.clients_lock:
            states = list(self.clients.values())
        for state in states:
            with state.lock:
                state.filter_config = FilterConfig(**loaded.filter_config.to_dict())
        self._emit("model_loaded", loaded.describe())
        self._emit("log", f"Loaded model: {loaded.path.name}")
        return loaded

    def clear_model(self) -> None:
        if self.running.is_set():
            raise RuntimeError("Stop the server before clearing the loaded model")
        with self.model_lock:
            self.model = None
        self._emit("model_loaded", [])
        self._emit("log", "Machine-learning model cleared")

    def set_filter(self, subject_id: str, config: FilterConfig) -> None:
        config.validate()
        with self.clients_lock:
            state = self.clients.get(subject_id)
        if state is None:
            raise KeyError(f"Subject {subject_id} is not connected")
        with state.lock:
            state.filter_config = config
        self._emit("subject_updated", subject_id)

    def send_medical_alert(self, subject_id: str, message: str) -> None:
        with self.clients_lock:
            state = self.clients.get(subject_id)
        if state is None:
            raise KeyError(f"Subject {subject_id} is not connected")
        text = str(message or "").strip()
        if not text:
            text = "The medical team requests your immediate attention. Please contact the monitoring center."
        send_frame(
            state.socket,
            {
                "type": "medical_alert",
                "subject_id": subject_id,
                "severity": "critical",
                "message": text,
                "server_time": datetime.now().isoformat(timespec="seconds"),
            },
            lock=state.send_lock,
        )
        self._emit("log", f"Medical alert sent to {subject_id}: {text}")
        self._emit("medical_alert_sent", subject_id)

    def connection_rows(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        with self.clients_lock:
            states = list(self.clients.values())
        rows = []
        for state in states:
            with state.lock:
                rows.append(
                    {
                        "id": state.subject_id,
                        "length": _duration(now - state.connected_at),
                        "ip": f"{state.address[0]}:{state.address[1]}",
                        "packages": state.packet_count,
                        "ping": _metric(state.ping_ms),
                        "jitter": _metric(state.jitter_ms),
                        "processing_latency": _metric(state.processing_latency_ms),
                        "quality": _connection_quality(state, now),
                    }
                )
        return sorted(rows, key=lambda row: row["id"])

    def person_rows(self) -> list[dict[str, Any]]:
        with self.clients_lock:
            states = list(self.clients.values())
        rows = []
        for state in states:
            with state.lock:
                rows.append(
                    {
                        "id": state.subject_id,
                        "metadata": dict(state.metadata),
                        "signal_time": _duration(state.last_signal_time),
                        "pulse": f"{state.last_bpm:.1f} bpm" if state.last_bpm else "n/a",
                    }
                )
        return sorted(rows, key=lambda row: row["id"])

    def subject_ids(self) -> list[str]:
        with self.clients_lock:
            return sorted(self.clients)

    def subject_snapshot(self, subject_id: str) -> dict[str, Any] | None:
        with self.clients_lock:
            state = self.clients.get(subject_id)
        if state is None:
            return None
        with state.lock:
            assessment = state.last_assessment
            return {
                "id": state.subject_id,
                "metadata": dict(state.metadata),
                "raw_ecg": None if state.raw_ecg is None else state.raw_ecg.copy(),
                "filtered_ecg": None if state.filtered_ecg is None else state.filtered_ecg.copy(),
                "acceleration": None if state.acceleration is None else state.acceleration.copy(),
                "ecg_fs": state.ecg_fs,
                "acc_fs": state.acc_fs,
                "signal_time": state.last_signal_time,
                "bpm": state.last_bpm,
                "assessment": assessment,
                "filter_config": state.filter_config,
                "history_duration": state.history.duration("ecg"),
                "history_summary": state.history.summary(),
                "processing_latency_ms": state.processing_latency_ms,
                "confusion_matrix": state.confusion_matrix.copy(),
                "confusion_classes": [1, 2, 3],
                "labelled_predictions": state.labelled_predictions,
            }

    def subject_window(
        self,
        subject_id: str,
        end_seconds: float,
        window_seconds: float,
    ) -> dict[str, Any] | None:
        with self.clients_lock:
            state = self.clients.get(subject_id)
        if state is None:
            return None
        duration = state.history.duration("ecg")
        end_seconds = max(0.0, min(float(end_seconds), duration))
        start_seconds = max(0.0, end_seconds - max(0.25, float(window_seconds)))
        # Warm the causal display filter with preceding history. Because it
        # never reads future samples, an event that has already been plotted
        # cannot disappear merely because another packet arrived.
        context_start = max(0.0, start_seconds - 10.0)
        ecg_context = state.history.read("ecg", context_start, end_seconds)
        if ecg_context is None:
            return None
        with state.lock:
            filter_config = state.filter_config
        try:
            filtered_context = apply_filter_causal(
                ecg_context.samples, ecg_context.fs, filter_config
            )
        except Exception:
            filtered_context = ecg_context.samples.astype(float, copy=True)
        offset = max(
            0,
            min(
                len(ecg_context.samples),
                int(round((start_seconds - ecg_context.start_seconds) * ecg_context.fs)),
            ),
        )
        raw = ecg_context.samples[offset:].copy()
        filtered = filtered_context[offset:].copy()
        actual_start = ecg_context.start_seconds + offset / ecg_context.fs
        acceleration = state.history.read("acc", start_seconds, end_seconds)
        return {
            "start_seconds": actual_start,
            "end_seconds": ecg_context.end_seconds,
            "duration_seconds": duration,
            "raw_ecg": raw,
            "filtered_ecg": filtered,
            "ecg_fs": ecg_context.fs,
            "ecg_units": ecg_context.units,
            "ecg_channels": ecg_context.channel_names,
            "acceleration": None if acceleration is None else acceleration.samples,
            "acc_fs": 0.0 if acceleration is None else acceleration.fs,
            "acc_units": [] if acceleration is None else acceleration.units,
            "acc_channels": [] if acceleration is None else acceleration.channel_names,
        }

    def _accept_loop(self) -> None:
        while self.running.is_set():
            try:
                assert self._listener is not None
                connection, address = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with self.clients_lock:
                full = len(self.clients) >= self.config.max_clients
            if full:
                try:
                    send_frame(connection, {"type": "server_error", "message": "Server is full"})
                finally:
                    connection.close()
                continue
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            thread = threading.Thread(
                target=self._client_loop,
                args=(connection, address),
                name=f"ecg-client-{address[0]}-{address[1]}",
                daemon=True,
            )
            thread.start()

    def _client_loop(self, connection: socket.socket, address: tuple[str, int]) -> None:
        state: ClientState | None = None
        try:
            connection.settimeout(8.0)
            hello, payload = recv_frame(connection)
            if payload or hello.get("type") != "hello":
                raise ProtocolError("The first message must be a payload-free hello")
            subject_id = _safe_subject_id(hello.get("subject_id"))
            metadata = hello.get("metadata")
            if not isinstance(metadata, dict):
                raise ProtocolError("Hello message has no metadata object")
            requested = float(hello.get("requested_interval_seconds", 2.0))
            requested = float(np.clip(requested, 1.0, 5.0))
            interval = self.config.interval_override_seconds or requested
            session_id = str(hello.get("session_id") or uuid.uuid4())
            with self.clients_lock:
                history = self.histories.get(session_id)
                if history is None:
                    history = DiskSignalHistory(subject_id)
                    self.histories[session_id] = history
            state = ClientState(
                subject_id=subject_id,
                address=address,
                socket=connection,
                session_id=session_id,
                metadata=metadata,
                interval_seconds=float(interval),
                history=history,
            )
            with self.model_lock:
                if self.model is not None:
                    state.filter_config = FilterConfig(**self.model.filter_config.to_dict())
                model_window_seconds = (
                    float(self.model.window_seconds) if self.model is not None else 10.0
                )
                model_label_source = (
                    _model_label_source(self.model) if self.model is not None else "consensus"
                )
            with self.clients_lock:
                previous = self.clients.get(subject_id)
                self.clients[subject_id] = state
            if previous:
                previous.connected = False
                self._close_client_socket(previous.socket)
            send_frame(
                connection,
                {
                    "type": "hello_ack",
                    "server": "ekg-monitor-server/2.0",
                    "interval_seconds": interval,
                    "next_sequence": max(1, int(hello.get("next_sequence", 1))),
                    "connection_id": str(uuid.uuid4()),
                    "model_window_seconds": model_window_seconds,
                    "model_label_source": model_label_source,
                },
                lock=state.send_lock,
            )
            connection.settimeout(max(12.0, interval * 4.0))
            self._emit("log", f"Connected {subject_id} from {address[0]}:{address[1]}")
            self._emit("clients_changed", None)
            while self.running.is_set() and state.connected:
                header, payload = recv_frame(connection)
                message_type = header.get("type")
                if message_type == "ping":
                    send_frame(
                        connection,
                        {"type": "pong", "nonce": int(header.get("nonce", 0))},
                        lock=state.send_lock,
                    )
                elif message_type == "data":
                    self._process_data(state, header, payload)
                elif message_type == "goodbye":
                    break
                else:
                    raise ProtocolError(f"Unsupported client message: {message_type}")
        except Exception as exc:
            if self.running.is_set():
                self._emit("log", f"Client {address[0]}:{address[1]} ended: {exc}")
                if state:
                    try:
                        send_frame(
                            connection,
                            {"type": "server_error", "message": str(exc)},
                            lock=state.send_lock,
                        )
                    except Exception:
                        pass
        finally:
            self._close_client_socket(connection)
            if state:
                state.connected = False
                with self.clients_lock:
                    if self.clients.get(state.subject_id) is state:
                        del self.clients[state.subject_id]
                self._emit("log", f"Disconnected {state.subject_id}")
                self._emit("clients_changed", None)

    def _process_data(
        self, state: ClientState, header: dict[str, Any], payload: bytes
    ) -> None:
        processing_started = time.perf_counter()
        if str(header.get("subject_id")) != state.subject_id:
            raise ProtocolError("Data subject does not match the authenticated subject")
        sequence = int(header.get("sequence", -1))
        if sequence < 1:
            raise ProtocolError("Invalid packet sequence")
        with state.lock:
            duplicate_end = state.acknowledged_sequences.get(sequence)
        if duplicate_end is not None:
            self._send_ack(state, sequence, duplicate_end)
            return
        descriptors = header.get("streams")
        if not isinstance(descriptors, list):
            raise ProtocolError("Data message has no signal descriptors")
        blocks = decode_signal_blocks(descriptors, payload)
        if "ecg" not in blocks:
            raise ProtocolError("Data packet contains no ECG stream")
        ecg = blocks["ecg"]
        acceleration_block = blocks.get("acc")
        if ecg.fs <= 0:
            raise ProtocolError("ECG sampling frequency must be positive")
        with self.model_lock:
            model = self.model
        if model is None:
            raise RuntimeError("No trained model is loaded")
        state.history.append(
            "ecg", ecg.samples, ecg.fs, ecg.channel_names, ecg.units
        )
        if acceleration_block is not None:
            state.history.append(
                "acc",
                acceleration_block.samples,
                acceleration_block.fs,
                acceleration_block.channel_names,
                acceleration_block.units,
            )
        now = time.monotonic()
        with state.lock:
            interarrival = now - state.last_packet_at if state.packet_count else None
            state.last_packet_at = now
            if interarrival is not None:
                sample = abs(interarrival - state.interval_seconds) * 1000.0
                state.jitter_ms = 0.8 * state.jitter_ms + 0.2 * sample
                state.last_interarrival_seconds = interarrival
            report = header.get("network_report") or {}
            reported_rtt = report.get("rtt_ms")
            if reported_rtt is not None:
                state.ping_ms = float(reported_rtt)
            state.packet_count += 1
            state.bytes_received += len(payload)
            state.last_signal_time = int(header.get("end_sample", 0)) / ecg.fs
            filter_config = state.filter_config
        with state.lock:
            previous_ecg = state.raw_ecg
            previous_acceleration = state.acceleration
        retention_seconds = max(10.0, float(model.window_seconds))
        raw = _rolling_append(
            previous_ecg,
            ecg.samples,
            int(round(ecg.fs * retention_seconds)),
        )
        acceleration = (
            _rolling_append(
                previous_acceleration,
                acceleration_block.samples,
                int(round(acceleration_block.fs * retention_seconds)),
            )
            if acceleration_block
            else previous_acceleration
        )
        try:
            filtered = apply_filter(raw, ecg.fs, filter_config)
        except Exception as exc:
            filtered = raw.astype(float, copy=True)
            self._throttled_feedback(state, "filter", "signal", "warning", f"Filter failed: {exc}")
        if len(filtered) < int(ecg.fs * 3.0):
            heuristic = SignalAssessment(
                quality_class=0,
                quality_label="warming up",
                quality_score=0.0,
                reason="Collecting at least three seconds for pulse and quality analysis",
            )
        else:
            heuristic = assess_signal(filtered[:, 0], ecg.fs)
        required_ecg_samples = max(2, int(round(model.window_seconds * ecg.fs)))
        if len(raw) < required_ecg_samples:
            assessment = SignalAssessment(
                quality_class=0,
                quality_label="warming up model",
                quality_score=0.0,
                reason=(
                    f"Collecting the exported {model.window_seconds:g}-second "
                    "model input window"
                ),
                bpm=heuristic.bpm,
                health_warning=heuristic.health_warning,
                source="machine learning",
                pulse_latest_rr_ms=heuristic.pulse_latest_rr_ms,
                pulse_interval_count=heuristic.pulse_interval_count,
                pulse_window_seconds=heuristic.pulse_window_seconds,
            )
        else:
            try:
                model_raw = raw[-required_ecg_samples:]
                # This is deliberately the same zero-phase, per-window filter
                # used by build_dataset(). Display filters never enter this path.
                model_ecg = apply_filter(
                    model_raw, ecg.fs, model.filter_config
                )
                model_acceleration = acceleration
                acceleration_fs = (
                    acceleration_block.fs
                    if acceleration_block is not None
                    else state.acc_fs
                )
                if model_acceleration is not None and acceleration_fs > 0:
                    required_acc_samples = max(
                        1, int(round(model.window_seconds * acceleration_fs))
                    )
                    model_acceleration = model_acceleration[-required_acc_samples:]
                predicted = model.predict(
                    model_ecg,
                    model_acceleration,
                    state.metadata,
                    acceleration_fs=acceleration_fs or 1.0,
                )
                predicted.bpm = heuristic.bpm
                predicted.health_warning = heuristic.health_warning
                predicted.pulse_latest_rr_ms = heuristic.pulse_latest_rr_ms
                predicted.pulse_interval_count = heuristic.pulse_interval_count
                predicted.pulse_window_seconds = heuristic.pulse_window_seconds
                assessment = predicted
            except Exception as exc:
                assessment = SignalAssessment(
                    quality_class=0,
                    quality_label="model inference unavailable",
                    quality_score=0.0,
                    reason=str(exc),
                    bpm=heuristic.bpm,
                    health_warning=heuristic.health_warning,
                    source="machine learning",
                    pulse_latest_rr_ms=heuristic.pulse_latest_rr_ms,
                    pulse_interval_count=heuristic.pulse_interval_count,
                    pulse_window_seconds=heuristic.pulse_window_seconds,
                )
                self._throttled_feedback(
                    state,
                    "model",
                    "signal",
                    "error",
                    f"Model inference failed: {exc}",
                )
        with state.lock:
            state.raw_ecg = raw
            state.filtered_ecg = filtered
            state.acceleration = acceleration
            state.ecg_fs = ecg.fs
            if acceleration_block is not None:
                state.acc_fs = acceleration_block.fs
            state.last_bpm = assessment.bpm
            state.last_assessment = assessment
            reference_class = int(header.get("reference_quality_class", 0) or 0)
            predicted_class = int(assessment.quality_class)
            if 1 <= reference_class <= 3 and 1 <= predicted_class <= 3:
                state.confusion_matrix[reference_class - 1, predicted_class - 1] += 1
                state.labelled_predictions += 1
            latency_sample_ms = (time.perf_counter() - processing_started) * 1000.0
            state.processing_latency_ms = (
                latency_sample_ms
                if state.processing_latency_ms is None
                else 0.8 * state.processing_latency_ms + 0.2 * latency_sample_ms
            )
            end_sample = int(header.get("end_sample", 0))
            state.acknowledged_sequences[sequence] = end_sample
            if len(state.acknowledged_sequences) > 512:
                for old_sequence in sorted(state.acknowledged_sequences)[:128]:
                    del state.acknowledged_sequences[old_sequence]
        if self.config.save_chunks:
            self._save_packet(state, sequence, blocks)
        self._send_ack(state, sequence, end_sample, assessment)
        if assessment.quality_class >= 2:
            severity = "error" if assessment.quality_class == 3 else "warning"
            self._throttled_feedback(
                state,
                "signal-quality",
                "signal noise",
                severity,
                f"Signal quality class {assessment.quality_class}: {assessment.reason}",
            )
        if assessment.health_warning:
            self._throttled_feedback(
                state, "health", "health", "error", assessment.health_warning
            )
        quality = _connection_quality(state, time.monotonic())
        if quality in {"fair", "poor"}:
            self._throttled_feedback(
                state,
                "connection-quality",
                "connection",
                "warning",
                f"Connection quality is {quality} (ping={_metric(state.ping_ms)}, jitter={_metric(state.jitter_ms)})",
                interval=15.0,
            )
        self._emit("subject_updated", state.subject_id)

    def _send_ack(
        self,
        state: ClientState,
        sequence: int,
        end_sample: int,
        assessment: SignalAssessment | None = None,
    ) -> None:
        header: dict[str, Any] = {
            "type": "data_ack",
            "sequence": sequence,
            "acknowledged_end_sample": end_sample,
            "server_time_ns": time.time_ns(),
        }
        if assessment:
            header.update(
                {
                    "quality_class": assessment.quality_class,
                    "quality_score": assessment.quality_score,
                    "pulse_bpm": assessment.bpm,
                }
            )
        send_frame(state.socket, header, lock=state.send_lock)

    def _throttled_feedback(
        self,
        state: ClientState,
        key: str,
        category: str,
        severity: str,
        message: str,
        interval: float = 8.0,
    ) -> None:
        now = time.monotonic()
        with state.lock:
            if now - state.feedback_times.get(key, 0.0) < interval:
                return
            state.feedback_times[key] = now
        send_frame(
            state.socket,
            {
                "type": "feedback",
                "category": category,
                "severity": severity,
                "message": message,
                "server_time": datetime.now().isoformat(timespec="seconds"),
            },
            lock=state.send_lock,
        )

    def _save_packet(self, state: ClientState, sequence: int, blocks: dict[str, Any]) -> None:
        safe_id = _safe_subject_id(state.subject_id)
        root = Path(self.config.save_root).expanduser().resolve()
        folder = (root / safe_id).resolve()
        if root != folder.parent:
            raise ValueError("Unsafe subject storage path")
        folder.mkdir(parents=True, exist_ok=True)
        arrays = {name: block.samples for name, block in blocks.items()}
        np.savez(folder / f"packet_{sequence:09d}.npz", **arrays)

    @staticmethod
    def _close_client_socket(connection: socket.socket) -> None:
        try:
            connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            connection.close()
        except OSError:
            pass

    def _emit(self, kind: str, payload: Any) -> None:
        self.events.put((kind, payload))


def _safe_subject_id(value: Any) -> str:
    subject_id = str(value or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", subject_id):
        raise ProtocolError("Invalid subject ID")
    return subject_id


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f} ms"


def _model_label_source(model: LoadedModel) -> str:
    training_config = model.metadata.get("training_config", {})
    dataset = training_config.get("dataset", {}) if isinstance(training_config, dict) else {}
    source = str(dataset.get("label_source", "consensus")).strip().lower()
    return source if source in {
        "consensus", "majority", "annotator1", "annotator2", "annotator3"
    } else "consensus"


def _connection_quality(state: ClientState, now: float) -> str:
    if now - state.last_packet_at > state.interval_seconds * 2.5:
        return "poor"
    ping = state.ping_ms or 0.0
    jitter = state.jitter_ms
    if ping > 400 or jitter > 300:
        return "poor"
    if ping > 200 or jitter > 150:
        return "fair"
    if ping > 80 or jitter > 60:
        return "good"
    return "excellent"


def _metadata_summary(metadata: dict[str, Any]) -> str:
    keys = ("gender", "age", "height", "weight", "smoker")
    values = [f"{key}={metadata[key]}" for key in keys if metadata.get(key) is not None]
    return ", ".join(values) if values else json.dumps(metadata, ensure_ascii=False)[:250]


def _rolling_append(
    previous: np.ndarray | None, current: np.ndarray, maximum_rows: int
) -> np.ndarray:
    values = np.asarray(current)
    if previous is not None and np.asarray(previous).ndim == values.ndim:
        values = np.concatenate((np.asarray(previous), values), axis=0)
    return values[-maximum_rows:] if len(values) > maximum_rows else values
