from __future__ import annotations

import argparse

from ekg_monitor.presentation.client_gui import ClientApp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Holter client simulator.")
    parser.add_argument("--dataset", default="butqdb", help="BUT QDB root folder.")
    parser.add_argument("--subject", help="Record / health-registry ID to select.")
    parser.add_argument("--host", default="127.0.0.1", help="Monitoring server address.")
    parser.add_argument("--port", type=int, default=50070, help="Monitoring server port.")
    parser.add_argument(
        "--interval", type=float, default=2.0, help="Packet interval from 1 to 5 seconds."
    )
    parser.add_argument(
        "--connect", action="store_true", help="Connect automatically after loading the record."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ClientApp(
        dataset_root=args.dataset,
        subject_id=args.subject,
        host=args.host,
        port=args.port,
        interval_seconds=args.interval,
        auto_start=args.connect,
    ).run()


if __name__ == "__main__":
    main()
