"""Run one local client and server with the saved model and one BUT QDB record."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ekg_monitor.application.client_service import ClientStreamingService
from ekg_monitor.application.models import ClientStreamConfig
from ekg_monitor.application.server_service import ServerConfig, ServerService

MODEL_SHA256 = "0b0adf27512aa4812d43f4fcc9214d7e8faed6a321c635a0b30a75925cda45df"


def replay(dataset: Path, record: str, timeout: float) -> dict:
    model_path = ROOT / "artifacts" / "model_bundle.joblib"
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != MODEL_SHA256:
        raise ValueError("The model checksum differs from the paper snapshot.")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    client = ClientStreamingService()
    server = ServerService()
    model = server.import_model(model_path)
    try:
        server.start(ServerConfig(host="127.0.0.1", port=port, save_chunks=False))
        client.start(ClientStreamConfig(dataset_root=str(dataset), subject_id=record,
                                        host="127.0.0.1", port=port,
                                        interval_seconds=2.0, auto_reconnect=False))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            snapshot = server.subject_snapshot(record)
            if snapshot and snapshot["assessment"] is not None:
                assessment = snapshot["assessment"]
                if (assessment.source == "machine learning"
                        and assessment.quality_class in (1, 2, 3)
                        and snapshot["history_duration"] >= model.window_seconds):
                    return {
                        "record": record,
                        "connection": "IPv4 loopback",
                        "server_history_seconds": snapshot["history_duration"],
                        "configured_packet_seconds": 2.0,
                        "model_window_seconds": model.window_seconds,
                        "feature_count": len(model.feature_columns),
                        "predicted_quality_class": int(assessment.quality_class),
                        "model_sha256": MODEL_SHA256,
                    }
            time.sleep(0.1)
        raise RuntimeError(f"No valid model prediction received within {timeout:g} seconds.")
    finally:
        client.stop()
        server.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Extracted BUT QDB 1.0.0 folder containing record directories.")
    parser.add_argument("--record", default="100001")
    parser.add_argument("--timeout", default=40.0, type=float)
    args = parser.parse_args()
    print(json.dumps(replay(args.dataset, args.record, args.timeout), indent=2))
