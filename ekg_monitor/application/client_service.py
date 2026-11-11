from __future__ import annotations

import queue
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from ekg_monitor.infrastructure.butqdb_repository import ButqdbRepository, interval_label
from ekg_monitor.infrastructure.protocol import (
    SignalBlock,
    encode_signal_blocks,
    recv_frame,
    send_frame,
)

from .models import ClientStreamConfig


@dataclass(slots=True)
class _PendingFrame:
    header: dict[str, Any]
    payload: bytes
    end_sample: int
    last_sent: float
    attempts: int = 1


class ClientStreamingService:
    """Reliable, reconnecting BUT QDB Holter simulator."""

    def __init__(self) -> None:
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._receiver: threading.Thread | None = None
        self._stop = threading.Event()
        self._disconnected = threading.Event()
        self._socket: socket.socket | None = None
        self._socket_lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending: dict[int, _PendingFrame] = {}
        self._last_ack_sample = 0
        self._last_rtt_ms: float | None = None
        self._next_sequence = 1
        self._session_id = str(uuid.uuid4())

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive() and not self._stop.is_set())

    def start(self, config: ClientStreamConfig) -> None:
        config.validate()
        if self.running:
            raise RuntimeError("Streaming is already active")
        self._stop.clear()
        self._disconnected.clear()
        self._last_ack_sample = 0
        self._next_sequence = 1
        with self._pending_lock:
            self._pending.clear()
        self._thread = threading.Thread(
            target=self._run, args=(config,), name="holter-stream", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._close_socket()
        self._emit("status", {"state": "stopped", "message": "Streaming stopped"})

    def _run(self, config: ClientStreamConfig) -> None:
        repository = ButqdbRepository()
        try:
            repository.scan(config.dataset_root)
            record = repository.get(config.subject_id)
            if record.ecg_header is None:
                raise ValueError(f"Record {config.subject_id} does not contain ECG data")
        except Exception as exc:
            self._emit_error(f"Cannot load record: {exc}")
            return
        self._emit("metadata", record.connection_metadata())
        failures = 0
        while not self._stop.is_set():
            try:
                self._stream_connection(config, repository, record)
                if self._last_ack_sample >= record.ecg_header.sample_count:
                    self._emit(
                        "status",
                        {"state": "complete", "message": "The complete record was streamed"},
                    )
                    break
                if self._stop.is_set():
                    break
                raise ConnectionError("Connection ended before the record was complete")
            except Exception as exc:
                with self._pending_lock:
                    if self._pending:
                        self._next_sequence = min(self._pending)
                self._close_socket()
                if self._stop.is_set():
                    break
                failures += 1
                self._emit_feedback("connection", "warning", f"Connection lost: {exc}")
                if not config.auto_reconnect or (
                    config.reconnect_limit > 0 and failures > config.reconnect_limit
                ):
                    self._emit_error("Reconnect limit reached")
                    break
                delay = min(30.0, float(2 ** min(failures - 1, 5)))
                self._emit(
                    "status",
                    {"state": "reconnecting", "message": f"Retrying in {delay:g} s"},
                )
                if self._stop.wait(delay):
                    break
        self._close_socket()
        if not self._stop.is_set() and self._last_ack_sample < record.ecg_header.sample_count:
            self._emit("status", {"state": "disconnected", "message": "Disconnected"})

    def _stream_connection(self, config, repository, record) -> None:
        self._emit(
            "status",
            {"state": "connecting", "message": f"Connecting to {config.host}:{config.port}"},
        )
        connection = socket.create_connection((config.host, config.port), timeout=6.0)
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        connection.settimeout(12.0)
        with self._socket_lock:
            self._socket = connection
        hello = {
            "type": "hello",
            "protocol": 1,
            "client": "butqdb-holter-simulator/2.0",
            "session_id": self._session_id,
            "subject_id": record.record_id,
            "requested_interval_seconds": config.interval_seconds,
            "resume_from_sample": self._last_ack_sample,
            "next_sequence": self._next_sequence,
            "metadata": record.connection_metadata(),
        }
        send_frame(connection, hello, lock=self._send_lock)
        response, _ = recv_frame(connection)
        if response.get("type") == "server_error":
            raise ConnectionError(str(response.get("message", "Server rejected the connection")))
        if response.get("type") != "hello_ack":
            raise ConnectionError("Server did not acknowledge the handshake")
        effective_interval = float(response.get("interval_seconds", config.interval_seconds))
        if not 1.0 <= effective_interval <= 5.0:
            raise ConnectionError("Server selected an invalid streaming interval")
        model_window_seconds = float(response.get("model_window_seconds", 10.0))
        if model_window_seconds <= 0:
            raise ConnectionError("Server selected an invalid model window")
        model_label_source = str(
            response.get("model_label_source", "consensus")
        ).strip().lower()
        annotation_groups = repository.annotation_groups(record.record_id)
        connection.settimeout(1.0)
        self._disconnected.clear()
        with self._pending_lock:
            self._pending.clear()
        self._emit_feedback(
            "connection",
            "info",
            f"Connected; server selected a {effective_interval:g} s interval",
        )
        self._emit("status", {"state": "streaming", "message": "Streaming"})
        self._receiver = threading.Thread(
            target=self._receive_loop,
            args=(connection,),
            name="holter-feedback",
            daemon=True,
        )
        self._receiver.start()

        next_sample = self._last_ack_sample
        sequence = max(self._next_sequence, int(response.get("next_sequence", 1)))
        send_deadline = time.monotonic()
        last_ping = 0.0
        while not self._stop.is_set() and not self._disconnected.is_set():
            now = time.monotonic()
            self._retry_expired(connection, effective_interval, now)
            if now - last_ping >= max(1.0, effective_interval):
                send_frame(
                    connection,
                    {"type": "ping", "nonce": time.monotonic_ns()},
                    lock=self._send_lock,
                )
                last_ping = now
            if now < send_deadline:
                self._stop.wait(min(0.1, send_deadline - now))
                continue
            with self._pending_lock:
                pending_count = len(self._pending)
            if pending_count >= 3:
                self._stop.wait(0.05)
                continue
            header_info = record.ecg_header
            if next_sample >= header_info.sample_count:
                while self._pending and not self._stop.is_set() and not self._disconnected.is_set():
                    self._retry_expired(connection, effective_interval, time.monotonic())
                    self._stop.wait(0.05)
                if not self._disconnected.is_set():
                    send_frame(connection, {"type": "goodbye"}, lock=self._send_lock)
                return
            start_seconds = next_sample / header_info.fs
            duration = min(
                effective_interval,
                (header_info.sample_count - next_sample) / header_info.fs,
            )
            window = repository.read_time_window(record.record_id, start_seconds, duration)
            blocks: list[SignalBlock] = []
            for name in ("ecg", "acc"):
                data = window.get(name)
                if not data:
                    continue
                blocks.append(
                    SignalBlock(
                        name=name,
                        samples=data["samples"],
                        fs=float(data["fs"]),
                        channel_names=list(data["channel_names"]),
                        units=list(data["units"]),
                    )
                )
            descriptors, payload = encode_signal_blocks(blocks)
            ecg_end = int(window["ecg"]["end_sample"])
            reference_start = max(
                0,
                ecg_end - int(round(model_window_seconds * header_info.fs)),
            )
            reference_quality_class = interval_label(
                annotation_groups,
                reference_start,
                ecg_end,
                model_label_source,
            )
            header = {
                "type": "data",
                "subject_id": record.record_id,
                "sequence": sequence,
                "start_sample": next_sample,
                "end_sample": ecg_end,
                "signal_time_seconds": start_seconds,
                "sent_time_ns": time.time_ns(),
                "reference_quality_class": int(reference_quality_class),
                "reference_label_source": model_label_source,
                "streams": descriptors,
                "network_report": {"rtt_ms": self._last_rtt_ms},
            }
            send_frame(connection, header, payload, lock=self._send_lock)
            sent_at = time.monotonic()
            with self._pending_lock:
                self._pending[sequence] = _PendingFrame(
                    header=header,
                    payload=payload,
                    end_sample=ecg_end,
                    last_sent=sent_at,
                )
            self._emit(
                "preview",
                {
                    "sequence": sequence,
                    "start_sample": next_sample,
                    "end_sample": ecg_end,
                    "total_samples": header_info.sample_count,
                    "signal_time_seconds": start_seconds,
                    "reference_quality_class": int(reference_quality_class),
                    "signals": {name: data["samples"] for name, data in window.items()},
                    "fs": {name: float(data["fs"]) for name, data in window.items()},
                },
            )
            next_sample = ecg_end
            sequence += 1
            self._next_sequence = sequence
            send_deadline += effective_interval
            if send_deadline < time.monotonic() - effective_interval:
                send_deadline = time.monotonic() + effective_interval
        if self._disconnected.is_set() and not self._stop.is_set():
            raise ConnectionError("Server stopped responding")

    def _retry_expired(self, connection: socket.socket, interval: float, now: float) -> None:
        timeout = max(5.0, interval * 2.5)
        resend: list[_PendingFrame] = []
        with self._pending_lock:
            for pending in self._pending.values():
                if now - pending.last_sent < timeout:
                    continue
                if pending.attempts >= 3:
                    self._disconnected.set()
                    raise ConnectionError("Server did not acknowledge data after three attempts")
                pending.attempts += 1
                pending.last_sent = now
                resend.append(pending)
        for pending in resend:
            send_frame(connection, pending.header, pending.payload, lock=self._send_lock)
            self._emit_feedback(
                "connection",
                "warning",
                f"Resent unacknowledged packet {pending.header['sequence']}",
            )

    def _receive_loop(self, connection: socket.socket) -> None:
        try:
            while not self._stop.is_set() and not self._disconnected.is_set():
                try:
                    header, _ = recv_frame(connection)
                except TimeoutError:
                    continue
                message_type = header.get("type")
                if message_type == "data_ack":
                    sequence = int(header.get("sequence", -1))
                    with self._pending_lock:
                        pending = self._pending.pop(sequence, None)
                        if pending:
                            self._last_ack_sample = max(self._last_ack_sample, pending.end_sample)
                    self._emit("ack", header)
                elif message_type == "pong":
                    nonce = int(header.get("nonce", 0))
                    if nonce:
                        self._last_rtt_ms = max(0.0, (time.monotonic_ns() - nonce) / 1_000_000)
                        self._emit("network", {"rtt_ms": self._last_rtt_ms})
                elif message_type == "feedback":
                    self._emit_feedback(
                        str(header.get("category", "server")),
                        str(header.get("severity", "info")),
                        str(header.get("message", "")),
                    )
                elif message_type == "medical_alert":
                    alert = {
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "category": "MEDICAL ALERT",
                        "severity": "critical",
                        "message": str(header.get("message", "Medical alert from the monitoring center")),
                    }
                    self._emit("feedback", alert)
                    self._emit("medical_alert", alert)
                elif message_type == "server_error":
                    raise ConnectionError(str(header.get("message", "Server error")))
        except Exception as exc:
            if not self._stop.is_set():
                self._emit_feedback("connection", "warning", str(exc))
            self._disconnected.set()

    def _close_socket(self) -> None:
        with self._socket_lock:
            connection, self._socket = self._socket, None
        if connection is not None:
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

    def _emit_error(self, message: str) -> None:
        self._emit_feedback("connection", "error", message)
        self._emit("status", {"state": "error", "message": message})

    def _emit_feedback(self, category: str, severity: str, message: str) -> None:
        self._emit(
            "feedback",
            {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "category": category,
                "severity": severity,
                "message": message,
            },
        )
