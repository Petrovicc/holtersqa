from __future__ import annotations

import argparse
import queue
import time
from pathlib import Path

from ekg_monitor.application.client_service import ClientStreamingService
from ekg_monitor.application.models import ClientStreamConfig
from ekg_monitor.infrastructure.butqdb_repository import ButqdbRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream any number of BUT QDB subjects in parallel."
    )
    parser.add_argument(
        "subjects",
        nargs="*",
        help="Record IDs separated by spaces or commas, for example 100001 103001.",
    )
    parser.add_argument("--dataset", default="butqdb", help="BUT QDB root folder.")
    parser.add_argument("--host", default="127.0.0.1", help="Monitoring server address.")
    parser.add_argument("--port", type=int, default=50070, help="Monitoring server port.")
    parser.add_argument(
        "--interval", type=float, default=2.0, help="Requested packet interval, 1 to 5 seconds."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Use the first N database records when explicit subject IDs are omitted.",
    )
    parser.add_argument(
        "--all", action="store_true", help="Stream every ECG record in the database."
    )
    parser.add_argument(
        "--stagger",
        type=float,
        default=0.1,
        help="Delay between client starts in seconds.",
    )
    parser.add_argument(
        "--no-reconnect", action="store_true", help="Stop a subject after a lost connection."
    )
    return parser.parse_args()


def selected_subjects(args: argparse.Namespace, repository: ButqdbRepository) -> list[str]:
    available = [
        record.record_id for record in repository.records.values() if record.ecg_header
    ]
    explicit = [
        item.strip()
        for argument in args.subjects
        for item in argument.split(",")
        if item.strip()
    ]
    if explicit:
        missing = sorted(set(explicit) - set(available))
        if missing:
            raise ValueError(f"Unknown record IDs: {', '.join(missing)}")
        return list(dict.fromkeys(explicit))
    if args.all:
        return available
    if args.count > 0:
        return available[: args.count]
    raise ValueError("Provide record IDs, --count N, or --all")


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset).expanduser().resolve()
    repository = ButqdbRepository(dataset)
    subjects = selected_subjects(args, repository)
    clients = {subject_id: ClientStreamingService() for subject_id in subjects}
    print(
        f"Starting {len(clients)} parallel subjects on {args.host}:{args.port}: "
        + ", ".join(subjects)
    )
    try:
        for subject_id, client in clients.items():
            client.start(
                ClientStreamConfig(
                    dataset_root=str(dataset),
                    subject_id=subject_id,
                    host=args.host,
                    port=args.port,
                    interval_seconds=args.interval,
                    auto_reconnect=not args.no_reconnect,
                )
            )
            if args.stagger > 0:
                time.sleep(args.stagger)
        while any(client.running for client in clients.values()):
            for subject_id, client in clients.items():
                while True:
                    try:
                        kind, payload = client.events.get_nowait()
                    except queue.Empty:
                        break
                    if kind == "status":
                        print(f"[{subject_id}] {payload['state']}: {payload['message']}")
                    elif kind == "feedback" and payload.get("severity") in {
                        "warning",
                        "error",
                        "critical",
                    }:
                        print(
                            f"[{subject_id}] {payload['severity']} "
                            f"{payload['category']}: {payload['message']}"
                        )
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping all subjects...")
    finally:
        for client in clients.values():
            client.stop()


if __name__ == "__main__":
    main()
