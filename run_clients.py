from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from ekg_monitor.infrastructure.butqdb_repository import ButqdbRepository
from run_subject import selected_subjects


ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# EDIT THESE VALUES, THEN RUN:  python run_clients.py
# ---------------------------------------------------------------------------
CLIENT_COUNT = 3
SUBJECT_IDS: tuple[str, ...] = ()  # Optional exact IDs; overrides CLIENT_COUNT.
DATASET_FOLDER = ROOT / "butqdb"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 50070
PACKET_INTERVAL_SECONDS = 2.0
WINDOW_START_DELAY_SECONDS = 0.25
AUTO_CONNECT = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open an independent Holter client window for every selected subject."
    )
    parser.add_argument("subjects", nargs="*", help="Record IDs separated by spaces or commas.")
    parser.add_argument("--dataset", default=str(DATASET_FOLDER), help="BUT QDB root folder.")
    parser.add_argument("--host", default=SERVER_HOST, help="Monitoring server address.")
    parser.add_argument("--port", type=int, default=SERVER_PORT, help="Monitoring server port.")
    parser.add_argument(
        "--interval",
        type=float,
        default=PACKET_INTERVAL_SECONDS,
        help="Packet interval, 1 to 5 seconds.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Open the first N records instead of CLIENT_COUNT.",
    )
    parser.add_argument("--all", action="store_true", help="Open every ECG record.")
    parser.add_argument(
        "--stagger",
        type=float,
        default=WINDOW_START_DELAY_SECONDS,
        help="Delay between windows.",
    )
    connection = parser.add_mutually_exclusive_group()
    connection.add_argument(
        "--connect",
        dest="no_connect",
        action="store_false",
        help="Connect each window automatically.",
    )
    connection.add_argument(
        "--no-connect",
        dest="no_connect",
        action="store_true",
        help="Load each selected record but do not connect automatically.",
    )
    parser.set_defaults(no_connect=not AUTO_CONNECT)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the commands without opening windows."
    )
    return parser.parse_args()


def client_command(args: argparse.Namespace, dataset: Path, subject_id: str) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "run_client.py"),
        "--dataset",
        str(dataset),
        "--subject",
        subject_id,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--interval",
        str(args.interval),
    ]
    if not args.no_connect:
        command.append("--connect")
    return command


def main() -> None:
    args = parse_args()
    if not 1.0 <= args.interval <= 5.0:
        raise SystemExit("--interval must be between 1 and 5 seconds")
    if args.stagger < 0:
        raise SystemExit("--stagger cannot be negative")
    dataset = Path(args.dataset).expanduser().resolve()
    repository = ButqdbRepository(dataset)
    configured_subjects = list(SUBJECT_IDS)
    requested_subjects = args.subjects or configured_subjects
    requested_count = (
        args.count
        if args.count is not None
        else (0 if requested_subjects or args.all else CLIENT_COUNT)
    )
    selection = SimpleNamespace(
        subjects=requested_subjects,
        all=args.all,
        count=requested_count,
    )
    try:
        subjects = selected_subjects(selection, repository)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    commands = [client_command(args, dataset, subject_id) for subject_id in subjects]
    print(f"Opening {len(commands)} Holter client window(s): {', '.join(subjects)}")
    if args.dry_run:
        for command in commands:
            print(subprocess.list2cmdline(command))
        return

    processes: list[subprocess.Popen] = []
    try:
        for command in commands:
            processes.append(subprocess.Popen(command, cwd=ROOT))
            if args.stagger:
                time.sleep(args.stagger)
        while any(process.poll() is None for process in processes):
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("Closing all client windows...")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()


if __name__ == "__main__":
    main()
