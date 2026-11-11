from __future__ import annotations

import json
import queue
from pathlib import Path
from typing import Any

import numpy as np

from ekg_monitor.application.client_service import ClientStreamingService
from ekg_monitor.application.models import ClientStreamConfig
from ekg_monitor.infrastructure.butqdb_repository import ButqdbRepository

from .common import ACCENT, DANGER, GOOD, MUTED, WARNING, configure_global_theme, replace_table_rows, safe_float, safe_int, timed_axis
from .native_dialogs import choose_directory


class ClientApp:
    def __init__(
        self,
        *,
        dataset_root: str | Path | None = None,
        subject_id: str | None = None,
        host: str = "127.0.0.1",
        port: int = 50070,
        interval_seconds: float = 2.0,
        auto_start: bool = False,
    ) -> None:
        import dearpygui.dearpygui as dpg

        self.dpg = dpg
        self.initial_dataset_root = None if dataset_root is None else str(dataset_root)
        self.initial_subject_id = None if subject_id is None else str(subject_id)
        self.initial_host = str(host)
        self.initial_port = int(port)
        self.initial_interval = float(interval_seconds)
        self.auto_start_pending = bool(auto_start)
        self.service = ClientStreamingService()
        self.repository = ButqdbRepository()
        self.feedback: list[dict[str, Any]] = []
        self.acknowledged = 0
        self._build()

    def _build(self) -> None:
        dpg = self.dpg
        dpg.create_context()
        self.theme = configure_global_theme(dpg)
        default_root = (
            Path(self.initial_dataset_root).expanduser()
            if self.initial_dataset_root
            else Path.cwd() / "butqdb"
        )
        with dpg.window(tag="client.main", label="HolterSQA Client Simulator"):
            with dpg.group(horizontal=True):
                dpg.add_text("HolterSQA Client Simulator", tag="client.heading", color=ACCENT)
                dpg.add_text("Disconnected", tag="client.status", color=MUTED)
                dpg.add_button(
                    tag="client.theme_toggle",
                    label="Light mode",
                    callback=lambda: self.theme.toggle("client.theme_toggle"),
                )
            dpg.add_text("", tag="client.alert_banner", color=DANGER, show=False, wrap=1100)
            dpg.add_separator()
            with dpg.group(horizontal=True):
                with dpg.child_window(width=425, height=-1, border=True):
                    dpg.add_text("BUT QDB RECORD", color=ACCENT)
                    dpg.add_input_text(
                        tag="client.dataset",
                        label="Dataset folder",
                        default_value=str(default_root) if default_root.is_dir() else "",
                        width=-120,
                    )
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Browse", callback=self._browse_dataset)
                        dpg.add_button(label="Scan records", callback=self._scan)
                    dpg.add_combo([], tag="client.subject", label="Record / registry ID", width=-1)
                    dpg.add_text("Select a record to display its complete metadata.", tag="client.meta_text", wrap=390)
                    dpg.add_separator()
                    dpg.add_text("SERVER CONNECTION", color=ACCENT)
                    dpg.add_input_text(tag="client.host", label="Server IP", default_value=self.initial_host)
                    dpg.add_input_int(tag="client.port", label="Port", default_value=self.initial_port, min_value=1, max_value=65535)
                    dpg.add_slider_float(
                        tag="client.interval",
                        label="Send every (seconds)",
                        default_value=self.initial_interval,
                        min_value=1.0,
                        max_value=5.0,
                        format="%.1f",
                    )
                    dpg.add_checkbox(tag="client.reconnect", label="Reconnect automatically", default_value=True)
                    with dpg.group(horizontal=True):
                        dpg.add_button(tag="client.start", label="Connect and stream", callback=self._start)
                        dpg.add_button(tag="client.stop", label="Stop", callback=self._stop, enabled=False)
                    dpg.add_progress_bar(tag="client.progress", default_value=0.0, overlay="0.0%", width=-1)
                    dpg.add_text("Packets acknowledged: 0", tag="client.packet_count")
                    dpg.add_text("Ping: n/a", tag="client.ping")
                with dpg.child_window(width=-1, height=-1, border=True):
                    with dpg.tab_bar():
                        with dpg.tab(label="Feedback from server"):
                            dpg.add_text(
                                "Connection, signal-noise, and health messages appear here.",
                                color=MUTED,
                            )
                            with dpg.table(
                                tag="client.feedback_table",
                                header_row=True,
                                row_background=True,
                                borders_innerH=True,
                                borders_outerH=True,
                                borders_innerV=True,
                                borders_outerV=True,
                                resizable=True,
                                scrollY=True,
                                height=-1,
                            ):
                                dpg.add_table_column(label="Time", width_fixed=True, init_width_or_weight=80)
                                dpg.add_table_column(label="Category", width_fixed=True, init_width_or_weight=120)
                                dpg.add_table_column(label="Severity", width_fixed=True, init_width_or_weight=80)
                                dpg.add_table_column(label="Message", width_stretch=True)
                        with dpg.tab(label="Live preview"):
                            with dpg.plot(label="ECG packet", height=-1, width=-1, tag="client.ecg_plot"):
                                dpg.add_plot_legend()
                                dpg.add_plot_axis(dpg.mvXAxis, label="Packet time (s)", tag="client.ecg_x")
                                with dpg.plot_axis(dpg.mvYAxis, label="ECG", tag="client.ecg_y"):
                                    dpg.add_line_series([], [], label="ECG", tag="client.ecg_series")
                        with dpg.tab(label="Metadata"):
                            with dpg.table(
                                tag="client.metadata_table",
                                header_row=True,
                                row_background=True,
                                borders_innerH=True,
                                borders_outerH=True,
                                borders_innerV=True,
                                borders_outerV=True,
                                resizable=True,
                            ):
                                dpg.add_table_column(label="Field", width_fixed=True, init_width_or_weight=180)
                                dpg.add_table_column(label="Value", width_stretch=True)
        with dpg.window(
            tag="client.alert_modal",
            label="Medical Alert",
            modal=True,
            show=False,
            no_resize=True,
            width=560,
            height=210,
        ):
            dpg.add_text("MEDICAL ALERT", tag="client.alert_heading", color=DANGER)
            dpg.add_separator()
            dpg.add_text("", tag="client.alert_message", wrap=520)
            dpg.add_spacer(height=8)
            dpg.add_button(
                label="Acknowledge",
                callback=lambda: dpg.configure_item("client.alert_modal", show=False),
                width=150,
            )
        self.theme.bind_heading("client.heading")
        self.theme.bind_heading("client.alert_heading")
        self.theme.bind_role("client.ecg_series", "raw")
        dpg.create_viewport(title="HolterSQA Client Simulator", width=1250, height=780, min_width=950, min_height=620)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("client.main", True)
        dpg.set_exit_callback(self._stop)
        if default_root.is_dir():
            self._scan()

    def run(self) -> None:
        dpg = self.dpg
        while dpg.is_dearpygui_running():
            self._poll_events()
            dpg.render_dearpygui_frame()
        self.service.stop()
        dpg.destroy_context()

    def _browse_dataset(self, *_args) -> None:
        path = choose_directory(
            "Select BUT QDB folder", self.dpg.get_value("client.dataset")
        )
        if path:
            self.dpg.set_value("client.dataset", path)
            self._scan()

    def _scan(self, *_args) -> None:
        dpg = self.dpg
        try:
            records = self.repository.scan(dpg.get_value("client.dataset"))
            ids = [record.record_id for record in records if record.ecg_header]
            dpg.configure_item("client.subject", items=ids, callback=self._subject_changed)
            if ids:
                selected = self.initial_subject_id if self.initial_subject_id in ids else ids[0]
                dpg.set_value("client.subject", selected)
                self._show_metadata(selected)
                if self.initial_subject_id and self.initial_subject_id not in ids:
                    self._local_feedback(
                        "dataset",
                        "error",
                        f"Requested record {self.initial_subject_id} was not found.",
                    )
                    self.auto_start_pending = False
            self._local_feedback("dataset", "info", f"Found {len(ids)} ECG records")
            if ids and self.auto_start_pending:
                self.auto_start_pending = False
                self._start()
        except Exception as exc:
            self._local_feedback("dataset", "error", str(exc))

    def _subject_changed(self, _sender, value) -> None:
        self._show_metadata(str(value))

    def _show_metadata(self, subject_id: str) -> None:
        try:
            metadata = self.repository.get(subject_id).connection_metadata()
        except Exception:
            return
        rows = [(key, json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value) for key, value in metadata.items()]
        replace_table_rows(self.dpg, "client.metadata_table", rows)
        summary = ", ".join(
            f"{key}: {metadata[key]}" for key in ("gender", "age", "height", "weight", "smoker") if metadata.get(key) is not None
        )
        self.dpg.set_value("client.meta_text", summary or f"Registry ID: {subject_id}")

    def _start(self, *_args) -> None:
        dpg = self.dpg
        try:
            config = ClientStreamConfig(
                dataset_root=dpg.get_value("client.dataset"),
                subject_id=dpg.get_value("client.subject"),
                host=dpg.get_value("client.host"),
                port=safe_int(dpg.get_value("client.port"), "Port"),
                interval_seconds=safe_float(dpg.get_value("client.interval"), "Interval"),
                auto_reconnect=bool(dpg.get_value("client.reconnect")),
            )
            self.acknowledged = 0
            self.service.start(config)
            dpg.configure_item("client.start", enabled=False)
            dpg.configure_item("client.stop", enabled=True)
        except Exception as exc:
            self._local_feedback("connection", "error", str(exc))

    def _stop(self, *_args) -> None:
        self.service.stop()
        if self.dpg.does_item_exist("client.start"):
            self.dpg.configure_item("client.start", enabled=True)
            self.dpg.configure_item("client.stop", enabled=False)

    def _poll_events(self) -> None:
        while True:
            try:
                kind, payload = self.service.events.get_nowait()
            except queue.Empty:
                break
            if kind == "status":
                state = payload["state"]
                color = GOOD if state in {"streaming", "complete"} else (WARNING if state in {"connecting", "reconnecting"} else DANGER)
                self.dpg.set_value("client.status", payload["message"])
                self.dpg.configure_item("client.status", color=color)
                if state in {"complete", "error", "disconnected", "stopped"}:
                    self.dpg.configure_item("client.start", enabled=True)
                    self.dpg.configure_item("client.stop", enabled=False)
            elif kind == "feedback":
                self.feedback.append(payload)
                self.feedback = self.feedback[-200:]
                self._refresh_feedback()
            elif kind == "medical_alert":
                message = str(
                    payload.get("message", "Medical alert from the monitoring center")
                )
                self.dpg.set_value("client.alert_message", message)
                self.dpg.set_value("client.alert_banner", f"MEDICAL ALERT: {message}")
                self.dpg.configure_item("client.alert_banner", show=True)
                self.dpg.configure_item("client.alert_modal", show=True)
            elif kind == "ack":
                self.acknowledged += 1
                self.dpg.set_value("client.packet_count", f"Packets acknowledged: {self.acknowledged}")
            elif kind == "network":
                self.dpg.set_value("client.ping", f"Ping: {payload['rtt_ms']:.1f} ms")
            elif kind == "preview":
                total = max(1, int(payload["total_samples"]))
                fraction = min(1.0, int(payload["end_sample"]) / total)
                self.dpg.set_value("client.progress", fraction)
                self.dpg.configure_item("client.progress", overlay=f"{fraction:.1%}")
                ecg = payload["signals"].get("ecg")
                if ecg is not None:
                    values = np.asarray(ecg)
                    x, y = timed_axis(values[:, 0] if values.ndim > 1 else values, payload["fs"]["ecg"])
                    self.dpg.set_value("client.ecg_series", [x, y])
                    self.dpg.fit_axis_data("client.ecg_x")
                    self.dpg.fit_axis_data("client.ecg_y")

    def _local_feedback(self, category: str, severity: str, message: str) -> None:
        from datetime import datetime

        self.feedback.append(
            {"timestamp": datetime.now().strftime("%H:%M:%S"), "category": category, "severity": severity, "message": message}
        )
        self.feedback = self.feedback[-200:]
        if self.dpg.does_item_exist("client.feedback_table"):
            self._refresh_feedback()

    def _refresh_feedback(self) -> None:
        replace_table_rows(
            self.dpg,
            "client.feedback_table",
            ((item["timestamp"], item["category"], item["severity"], item["message"]) for item in reversed(self.feedback)),
        )


def main() -> None:
    ClientApp().run()


if __name__ == "__main__":
    main()
