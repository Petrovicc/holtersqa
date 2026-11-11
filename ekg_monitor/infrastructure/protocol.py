from __future__ import annotations

import json
import socket
import struct
import threading
import zlib
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


MAGIC = b"EKG2"
PROTOCOL_VERSION = 1
_PREFIX = struct.Struct("!4sBIQI")
MAX_HEADER_BYTES = 2 * 1024 * 1024
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
ALLOWED_DTYPES = {"float32", "float64", "int16", "int32"}


class ProtocolError(ConnectionError):
    """Raised when a peer sends an invalid or corrupted frame."""


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        try:
            chunk = sock.recv(size - len(data))
        except socket.timeout as exc:
            raise TimeoutError("Timed out while receiving a frame") from exc
        if not chunk:
            raise ConnectionError("Peer closed the connection")
        data.extend(chunk)
    return bytes(data)


def encode_frame(header: dict[str, Any], payload: bytes = b"") -> bytes:
    if not isinstance(header, dict) or not header.get("type"):
        raise ValueError("Every frame requires a message type")
    header_bytes = json.dumps(
        header, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if not 0 < len(header_bytes) <= MAX_HEADER_BYTES:
        raise ValueError("Header is empty or too large")
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("Payload is too large")
    checksum = zlib.crc32(header_bytes)
    checksum = zlib.crc32(payload, checksum) & 0xFFFFFFFF
    return _PREFIX.pack(
        MAGIC, PROTOCOL_VERSION, len(header_bytes), len(payload), checksum
    ) + header_bytes + payload


def send_frame(
    sock: socket.socket,
    header: dict[str, Any],
    payload: bytes = b"",
    lock: threading.Lock | threading.RLock | None = None,
) -> None:
    frame = encode_frame(header, payload)
    if lock is None:
        sock.sendall(frame)
        return
    with lock:
        sock.sendall(frame)


def recv_frame(sock: socket.socket) -> tuple[dict[str, Any], bytes]:
    prefix = _recv_exact(sock, _PREFIX.size)
    magic, version, header_size, payload_size, expected_crc = _PREFIX.unpack(prefix)
    if magic != MAGIC:
        raise ProtocolError("Invalid frame magic")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"Unsupported protocol version {version}")
    if not 0 < header_size <= MAX_HEADER_BYTES:
        raise ProtocolError("Invalid header length")
    if payload_size > MAX_PAYLOAD_BYTES:
        raise ProtocolError("Invalid payload length")
    header_bytes = _recv_exact(sock, header_size)
    payload = _recv_exact(sock, payload_size) if payload_size else b""
    actual_crc = zlib.crc32(header_bytes)
    actual_crc = zlib.crc32(payload, actual_crc) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise ProtocolError("Frame checksum mismatch")
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("Invalid JSON header") from exc
    if not isinstance(header, dict) or not header.get("type"):
        raise ProtocolError("Missing message type")
    return header, payload


@dataclass(slots=True)
class SignalBlock:
    name: str
    samples: np.ndarray
    fs: float
    channel_names: list[str]
    units: list[str]


def encode_signal_blocks(blocks: Iterable[SignalBlock]) -> tuple[list[dict[str, Any]], bytes]:
    descriptors: list[dict[str, Any]] = []
    payload_parts: list[bytes] = []
    offset = 0
    for block in blocks:
        array = np.ascontiguousarray(block.samples, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        raw = array.tobytes(order="C")
        descriptors.append(
            {
                "name": block.name,
                "offset": offset,
                "nbytes": len(raw),
                "dtype": "float32",
                "shape": [int(v) for v in array.shape],
                "fs": float(block.fs),
                "channel_names": list(block.channel_names),
                "units": list(block.units),
            }
        )
        payload_parts.append(raw)
        offset += len(raw)
    return descriptors, b"".join(payload_parts)


def decode_signal_blocks(
    descriptors: list[dict[str, Any]], payload: bytes
) -> dict[str, SignalBlock]:
    result: dict[str, SignalBlock] = {}
    occupied: list[tuple[int, int]] = []
    for descriptor in descriptors:
        name = str(descriptor.get("name", "")).strip().lower()
        dtype_name = str(descriptor.get("dtype", ""))
        shape = descriptor.get("shape")
        offset = int(descriptor.get("offset", -1))
        nbytes = int(descriptor.get("nbytes", -1))
        if not name or name in result:
            raise ProtocolError("Signal block names must be unique and non-empty")
        if dtype_name not in ALLOWED_DTYPES:
            raise ProtocolError(f"Unsupported signal dtype: {dtype_name}")
        if not isinstance(shape, list) or len(shape) != 2:
            raise ProtocolError("Signal shape must contain two dimensions")
        rows, columns = int(shape[0]), int(shape[1])
        dtype = np.dtype(dtype_name)
        expected = rows * columns * dtype.itemsize
        if rows < 1 or columns < 1 or nbytes != expected:
            raise ProtocolError("Signal descriptor size does not match its shape")
        end = offset + nbytes
        if offset < 0 or end > len(payload):
            raise ProtocolError("Signal block points outside the payload")
        if any(not (end <= start0 or offset >= end0) for start0, end0 in occupied):
            raise ProtocolError("Signal blocks overlap")
        occupied.append((offset, end))
        array = np.frombuffer(payload[offset:end], dtype=dtype).reshape(rows, columns).copy()
        result[name] = SignalBlock(
            name=name,
            samples=array,
            fs=float(descriptor.get("fs", 0.0)),
            channel_names=[str(v) for v in descriptor.get("channel_names", [])],
            units=[str(v) for v in descriptor.get("units", [])],
        )
    if not result:
        raise ProtocolError("Data frame contains no signals")
    return result
