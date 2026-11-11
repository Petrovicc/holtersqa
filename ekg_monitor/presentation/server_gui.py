from __future__ import annotations

import json
import math
import queue
import time
from pathlib import Path
from typing import Any

import numpy as np

from ekg_monitor.application.models import FILTER_KINDS, FilterConfig
from ekg_monitor.application.server_service import ServerConfig, ServerService
from ekg_monitor.infrastructure.training_report_exporter import (
    render_confusion_matrix_array_rgba,
)

from .common import (
    ACCENT,
    DANGER,
    GOOD,
    MUTED,
    WARNING,
    configure_global_theme,
    replace_table_rows,
    safe_float,
    safe_int,
    timed_axis,
    to_millivolts,
)
from .native_dialogs import choose_directory, choose_open_file


SUBJECT_METADATA_COLUMNS: tuple[tuple[str, str, int], ...] = (
    ("health_registry_id", "Registry ID", 105),
    ("record_id", "Record ID", 95),
    ("person_id", "Person ID", 85),
    ("gender", "Gender", 70),
    ("age", "Age", 55),
    ("height", "Height (cm)", 90),
    ("weight", "Weight (kg)", 90),
    ("smoker", "Smoker", 75),
    ("ecg.record_name", "ECG record", 115),
    ("ecg.fs", "ECG rate (Hz)", 100),
    ("ecg.sample_count", "ECG samples", 110),
    ("ecg.signal_count", "ECG signals", 90),
    ("ecg.channel_names", "ECG channels", 125),
    ("ecg.units", "ECG units", 90),
    ("accelerometer.record_name", "ACC record", 115),
    ("accelerometer.fs", "ACC rate (Hz)", 100),
    ("accelerometer.sample_count", "ACC samples", 110),
    ("accelerometer.signal_count", "ACC signals", 90),
    ("accelerometer.channel_names", "ACC channels", 150),
    ("accelerometer.units", "ACC units", 110),
)


class ServerApp:
    def __init__(self) -> None:
        import dearpygui.dearpygui as dpg

        self.dpg = dpg
        self.service = ServerService()
        self.logs: list[str] = []
        self.subject_views: dict[str, dict[str, Any]] = {}
        self.last_table_refresh = 0.0
        self.last_subject_refresh = 0.0
        self._build()

    def _build(self) -> None:
        dpg = self.dpg
        dpg.create_context()
        with dpg.texture_registry(tag="server.subject_textures"):
            pass
        self.theme = configure_global_theme(dpg)
        with dpg.window(tag="server.main", label="ECG Monitoring Server"):
            with dpg.group(horizontal=True):
                dpg.add_text("ECG MONITORING SERVER", tag="server.heading", color=ACCENT)
                dpg.add_text("Stopped", tag="server.status", color=MUTED)
                dpg.add_button(
                    tag="server.theme_toggle",
                    label="Light mode",
                    callback=lambda: self.theme.toggle("server.theme_toggle"),
                )
            dpg.add_separator()
            with dpg.tab_bar(tag="server.tabs"):
                with dpg.tab(label="Connections", tag="server.connection_tab"):
                    self._connection_tab()
                with dpg.tab(label="Machine Learning", tag="server.ml_tab"):
                    self._ml_tab()
        self.theme.bind_heading("server.heading")
        dpg.create_viewport(
            title="ECG Monitoring Server",
            width=1500,
            height=940,
            min_width=1120,
            min_height=720,
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("server.main", True)
        dpg.set_exit_callback(self._stop)

    def _connection_tab(self) -> None:
        dpg = self.dpg
        with dpg.child_window(height=165, border=True):
            dpg.add_text("SERVER CONTROL", tag="server.control_heading", color=ACCENT)
            self.theme.bind_heading("server.control_heading")
            with dpg.group(horizontal=True):
                dpg.add_input_text(
                    tag="server.host", label="Bind address", default_value="0.0.0.0", width=180
                )
                dpg.add_input_int(
                    tag="server.port", label="Port", default_value=50070, width=135
                )
                dpg.add_input_int(
                    tag="server.max_clients", label="Maximum clients", default_value=16, width=145
                )
                dpg.add_combo(
                    ["Client choice", "1", "2", "3", "4", "5"],
                    tag="server.interval",
                    label="Packet interval (s)",
                    default_value="Client choice",
                    width=150,
                )
                dpg.add_button(
                    tag="server.start",
                    label="Start server",
                    callback=self._start,
                    enabled=False,
                )
                dpg.add_button(tag="server.stop", label="Stop server", callback=self._stop, enabled=False)
            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    tag="server.save_chunks", label="Archive received packets", default_value=False
                )
                dpg.add_input_text(
                    tag="server.save_root",
                    label="Archive folder",
                    default_value=str(Path.cwd() / "server_data_v2"),
                    width=420,
                )
                dpg.add_button(label="Browse", callback=self._choose_save_folder)
        with dpg.group(horizontal=True):
            with dpg.child_window(width=-1, height=270, border=True):
                dpg.add_text("LIVE CONNECTIONS", color=ACCENT)
                with dpg.table(
                    tag="server.connections_table",
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
                    for label in (
                        "ID",
                        "Connected",
                        "IP address",
                        "Packets",
                        "Ping",
                        "Jitter",
                        "Server processing latency",
                        "Quality",
                    ):
                        dpg.add_table_column(label=label)
        with dpg.child_window(height=235, border=True):
            dpg.add_text("CURRENT SUBJECT DATA", color=ACCENT)
            with dpg.table(
                tag="server.persons_table",
                header_row=True,
                row_background=True,
                borders_innerH=True,
                borders_outerH=True,
                borders_innerV=True,
                borders_outerV=True,
                resizable=True,
                height=-1,
                scrollY=True,
                scrollX=True,
                freeze_columns=1,
                policy=dpg.mvTable_SizingFixedFit,
            ):
                for _key, label, width in SUBJECT_METADATA_COLUMNS:
                    dpg.add_table_column(
                        label=label, width_fixed=True, init_width_or_weight=width
                    )
                dpg.add_table_column(label="Recording time", width_fixed=True, init_width_or_weight=145)
                dpg.add_table_column(label="Current pulse", width_fixed=True, init_width_or_weight=120)
        dpg.add_input_text(
            tag="server.log",
            label="Activity log",
            multiline=True,
            readonly=True,
            height=170,
            width=-1,
        )

    def _ml_tab(self) -> None:
        dpg = self.dpg
        with dpg.group(horizontal=True):
            dpg.add_text("MODEL WORKSPACE", tag="server.ml_heading", color=ACCENT)
            dpg.add_button(
                tag="server.import_model",
                label="Import trained model",
                callback=self._choose_model,
            )
            dpg.add_button(
                tag="server.clear_model",
                label="Clear model",
                callback=self._clear_model,
                enabled=False,
            )
        self.theme.bind_heading("server.ml_heading")
        dpg.add_text(
            "No model loaded. Import a current model artifact before the server can start.",
            tag="server.ml_summary",
            color=MUTED,
        )
        with dpg.table(header_row=False, policy=dpg.mvTable_SizingStretchProp):
            for _ in range(4):
                dpg.add_table_column()
            with dpg.table_row():
                for title, tag in (
                    ("MODEL TYPE", "server.ml_type"),
                    ("INPUTS", "server.ml_inputs"),
                    ("CLASSES", "server.ml_classes"),
                    ("COMPLEXITY", "server.ml_complexity"),
                ):
                    with dpg.table_cell():
                        with dpg.child_window(height=105, border=True):
                            dpg.add_text(title, color=MUTED)
                            dpg.add_text("-", tag=tag, wrap=300)
                            self.theme.bind_heading(tag)
        with dpg.child_window(height=150, border=True):
            dpg.add_text("PIPELINE", color=ACCENT)
            dpg.add_text("No pipeline available", tag="server.ml_pipeline", wrap=1350)
            dpg.add_text("ARTIFACT", color=ACCENT)
            dpg.add_text("No artifact selected", tag="server.ml_artifact", wrap=1350)
        with dpg.collapsing_header(label="Input feature definition", default_open=True):
            dpg.add_input_text(
                tag="server.ml_feature_names",
                multiline=True,
                readonly=True,
                height=120,
                width=-1,
                default_value="No inputs available",
            )
        with dpg.collapsing_header(label="Estimator parameters", default_open=True):
            with dpg.table(
                tag="server.ml_parameters",
                header_row=True,
                row_background=True,
                borders_innerH=True,
                borders_outerH=True,
                borders_innerV=True,
                borders_outerV=True,
                resizable=True,
                scrollY=True,
                height=260,
            ):
                dpg.add_table_column(label="Parameter", width_fixed=True, init_width_or_weight=300)
                dpg.add_table_column(label="Value", width_stretch=True)
        with dpg.collapsing_header(label="Training and artifact metadata"):
            with dpg.table(
                tag="server.ml_metadata",
                header_row=True,
                row_background=True,
                borders_innerH=True,
                borders_outerH=True,
                borders_innerV=True,
                borders_outerV=True,
                resizable=True,
                scrollY=True,
                height=230,
            ):
                dpg.add_table_column(label="Field", width_fixed=True, init_width_or_weight=300)
                dpg.add_table_column(label="Value", width_stretch=True)

    def _create_subject_tab(self, subject_id: str) -> None:
        if subject_id in self.subject_views:
            return
        dpg = self.dpg
        prefix = f"server.subject.{subject_id}"
        tab_tag = f"{prefix}.tab"
        with dpg.tab(
            label=f"Subject-{subject_id}", tag=tab_tag, parent="server.tabs", closable=False
        ):
            with dpg.group(horizontal=True):
                dpg.add_text(f"SUBJECT {subject_id}", tag=f"{prefix}.heading", color=ACCENT)
                dpg.add_text("Pulse: n/a", tag=f"{prefix}.pulse")
                dpg.add_text("Quality: waiting for signal", tag=f"{prefix}.quality", color=MUTED)
                dpg.add_text("Recording time: 00:00:00", tag=f"{prefix}.time")
                dpg.add_text("Processing latency: n/a", tag=f"{prefix}.latency")
                dpg.add_button(
                    label="Send Medical Alert",
                    tag=f"{prefix}.alert_button",
                    callback=self._alert_clicked,
                    user_data=subject_id,
                )
            dpg.add_input_text(
                tag=f"{prefix}.alert_text",
                label="Alert message",
                default_value="The medical team requests your immediate attention. Please contact the monitoring center.",
                width=-1,
            )
            dpg.add_text(
                "No medical alert sent in this session.",
                tag=f"{prefix}.alert_status",
                color=MUTED,
            )
            with dpg.child_window(height=165, border=True):
                with dpg.group(horizontal=True):
                    dpg.add_radio_button(
                        ["Raw", "Filtered", "Both"],
                        tag=f"{prefix}.signal_mode",
                        label="ECG signals",
                        default_value="Both",
                        horizontal=True,
                        callback=self._signal_mode_changed,
                        user_data=subject_id,
                    )
                    dpg.add_slider_float(
                        tag=f"{prefix}.window",
                        label="Visible window (s)",
                        default_value=10.0,
                        min_value=2.0,
                        max_value=60.0,
                        format="%.0f",
                        width=220,
                        callback=self._subject_display_changed,
                        user_data=subject_id,
                    )
                    dpg.add_checkbox(
                        tag=f"{prefix}.follow",
                        label="Follow latest",
                        default_value=True,
                    )
                    dpg.add_button(
                        label="Jump to latest",
                        tag=f"{prefix}.jump_latest",
                        callback=self._jump_latest_clicked,
                        user_data=subject_id,
                    )
                    dpg.add_combo(
                        ["25", "50"],
                        tag=f"{prefix}.speed",
                        label="Paper speed (mm/s)",
                        default_value="25",
                        width=90,
                        callback=self._subject_display_changed,
                        user_data=subject_id,
                    )
                    dpg.add_combo(
                        ["5", "10", "20"],
                        tag=f"{prefix}.gain",
                        label="Gain (mm/mV)",
                        default_value="10",
                        width=90,
                        callback=self._subject_display_changed,
                        user_data=subject_id,
                    )
                dpg.add_slider_float(
                    tag=f"{prefix}.timeline",
                    label="Timeline end (s)",
                    default_value=0.0,
                    min_value=0.0,
                    max_value=1.0,
                    width=-1,
                    callback=self._timeline_changed_callback,
                    user_data=subject_id,
                )
                dpg.add_text(
                    "Standard grid: small box 0.04 s x 0.1 mV; large box 0.2 s x 0.5 mV at 25 mm/s and 10 mm/mV.",
                    tag=f"{prefix}.calibration",
                    color=MUTED,
                )
            with dpg.collapsing_header(label="Signal filtration", default_open=False):
                with dpg.group(horizontal=True):
                    dpg.add_checkbox(tag=f"{prefix}.filter_dc", label="Remove DC", default_value=True)
                    dpg.add_combo(
                        list(FILTER_KINDS),
                        tag=f"{prefix}.filter_kind",
                        label="Filter",
                        default_value="butter_bandpass",
                        width=210,
                    )
                    dpg.add_input_int(tag=f"{prefix}.filter_order", label="Order/taps", default_value=4, width=110)
                    dpg.add_input_float(tag=f"{prefix}.filter_low", label="Low Hz", default_value=0.5, width=110)
                    dpg.add_input_float(tag=f"{prefix}.filter_high", label="High Hz", default_value=40.0, width=110)
                    dpg.add_input_float(tag=f"{prefix}.filter_ripple", label="Ripple dB", default_value=0.5, width=110)
                    dpg.add_button(
                        label="Apply filter",
                        tag=f"{prefix}.apply_filter",
                        callback=self._filter_clicked,
                        user_data=subject_id,
                    )
                dpg.add_text(
                    "Display/pulse filter only. ML detection always uses the immutable preprocessing exported with the model.",
                    tag=f"{prefix}.filter_message",
                    color=MUTED,
                )
            with dpg.plot(
                label="ECG monitor",
                height=430,
                width=-1,
                tag=f"{prefix}.ecg_plot",
                no_menus=True,
            ):
                dpg.add_plot_legend()
                dpg.add_plot_axis(
                    dpg.mvXAxis,
                    label="Time from start of monitoring (s)",
                    tag=f"{prefix}.ecg_x",
                    no_gridlines=True,
                )
                with dpg.plot_axis(
                    dpg.mvYAxis,
                    label="ECG (mV)",
                    tag=f"{prefix}.ecg_y",
                    no_gridlines=True,
                ):
                    dpg.add_inf_line_series([], tag=f"{prefix}.grid_v_minor")
                    dpg.add_inf_line_series([], tag=f"{prefix}.grid_v_major")
                    dpg.add_inf_line_series(
                        [], horizontal=True, tag=f"{prefix}.grid_h_minor"
                    )
                    dpg.add_inf_line_series(
                        [], horizontal=True, tag=f"{prefix}.grid_h_major"
                    )
                    dpg.add_line_series([], [], label="Raw ECG", tag=f"{prefix}.raw")
                    dpg.add_line_series([], [], label="Filtered ECG", tag=f"{prefix}.filtered")
            with dpg.plot(
                label="Accelerometer",
                height=230,
                width=-1,
                tag=f"{prefix}.acc_plot",
            ):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)", tag=f"{prefix}.acc_axis_x")
                with dpg.plot_axis(dpg.mvYAxis, label="Acceleration", tag=f"{prefix}.acc_axis_y"):
                    dpg.add_line_series([], [], label="ACC X", tag=f"{prefix}.acc_x")
                    dpg.add_line_series([], [], label="ACC Y", tag=f"{prefix}.acc_y")
                    dpg.add_line_series([], [], label="ACC Z", tag=f"{prefix}.acc_z")
            texture_tag = f"{prefix}.confusion_texture"
            dpg.add_dynamic_texture(
                620,
                540,
                np.zeros((540, 620, 4), dtype=np.float32).reshape(-1).tolist(),
                tag=texture_tag,
                parent="server.subject_textures",
            )
            with dpg.collapsing_header(
                label="Classification evaluation", default_open=True
            ):
                dpg.add_text(
                    "Updated every five labelled model windows. Rows are actual classes; columns are predictions; percentages are row-normalized.",
                    color=MUTED,
                    wrap=1350,
                )
                with dpg.group(horizontal=True):
                    dpg.add_image(texture_tag, width=620, height=540)
                    with dpg.child_window(width=-1, height=540, border=True):
                        dpg.add_text(
                            "No labelled classifications received yet.",
                            tag=f"{prefix}.confusion_summary",
                            wrap=620,
                        )
                        with dpg.table(
                            tag=f"{prefix}.confusion_metrics",
                            header_row=True,
                            row_background=True,
                            borders_innerH=True,
                            borders_outerH=True,
                            borders_innerV=True,
                            borders_outerV=True,
                        ):
                            for label in (
                                "Class", "Support", "Predicted", "Precision", "Recall", "F1"
                            ):
                                dpg.add_table_column(label=label)
            with dpg.collapsing_header(label="Subject metadata", default_open=True):
                with dpg.table(
                    tag=f"{prefix}.metadata",
                    header_row=True,
                    row_background=True,
                    borders_innerH=True,
                    borders_outerH=True,
                    borders_innerV=True,
                    borders_outerV=True,
                    resizable=True,
                    scrollX=True,
                    height=125,
                    freeze_columns=1,
                    policy=dpg.mvTable_SizingFixedFit,
                ):
                    for _key, label, width in SUBJECT_METADATA_COLUMNS:
                        dpg.add_table_column(
                            label=label, width_fixed=True, init_width_or_weight=width
                        )
        self.subject_views[subject_id] = {
            "prefix": prefix,
            "tab": tab_tag,
            "last_duration": -1.0,
            "texture": texture_tag,
            "confusion_refresh_bucket": -1,
        }
        self.theme.bind_heading(f"{prefix}.heading")
        self.theme.bind_role(f"{prefix}.alert_button", "danger_button")
        for tag, role in (
            (f"{prefix}.raw", "raw"),
            (f"{prefix}.filtered", "filtered"),
            (f"{prefix}.acc_x", "acc_x"),
            (f"{prefix}.acc_y", "acc_y"),
            (f"{prefix}.acc_z", "acc_z"),
            (f"{prefix}.grid_v_minor", "grid_minor"),
            (f"{prefix}.grid_h_minor", "grid_minor"),
            (f"{prefix}.grid_v_major", "grid_major"),
            (f"{prefix}.grid_h_major", "grid_major"),
        ):
            self.theme.bind_role(tag, role)
        snapshot = self.service.subject_snapshot(subject_id)
        if snapshot:
            self._load_subject_settings(subject_id, snapshot)
            self._render_subject_metadata(subject_id, snapshot["metadata"])
        dpg.set_value("server.tabs", tab_tag)

    def run(self) -> None:
        dpg = self.dpg
        while dpg.is_dearpygui_running():
            self._poll_events()
            now = time.monotonic()
            if now - self.last_table_refresh >= 0.5:
                self._refresh_tables()
                self.last_table_refresh = now
            if now - self.last_subject_refresh >= 0.25:
                for subject_id in list(self.subject_views):
                    self._refresh_subject_view(subject_id)
                self.last_subject_refresh = now
            dpg.render_dearpygui_frame()
        self.service.stop()
        dpg.destroy_context()

    def _start(self, *_args) -> None:
        dpg = self.dpg
        try:
            interval_text = dpg.get_value("server.interval")
            config = ServerConfig(
                host=dpg.get_value("server.host"),
                port=safe_int(dpg.get_value("server.port"), "Port"),
                max_clients=safe_int(dpg.get_value("server.max_clients"), "Maximum clients"),
                interval_override_seconds=(
                    0.0
                    if interval_text == "Client choice"
                    else safe_float(interval_text, "Interval")
                ),
                save_chunks=bool(dpg.get_value("server.save_chunks")),
                save_root=dpg.get_value("server.save_root"),
            )
            self.service.start(config)
            dpg.configure_item("server.start", enabled=False)
            dpg.configure_item("server.stop", enabled=True)
        except Exception as exc:
            self._append_log(f"START ERROR: {exc}")

    def _stop(self, *_args) -> None:
        self.service.stop()
        if self.dpg.does_item_exist("server.start"):
            self.dpg.configure_item(
                "server.start", enabled=self.service.model is not None
            )
            self.dpg.configure_item("server.stop", enabled=False)
            self.dpg.configure_item("server.import_model", enabled=True)
            self.dpg.configure_item(
                "server.clear_model", enabled=self.service.model is not None
            )

    def _choose_model(self, *_args) -> None:
        path = choose_open_file(
            "Import trained ECG model",
            (("Joblib model", "*.joblib"), ("Pickle model", "*.pkl"), ("All files", "*.*")),
            Path.cwd(),
        )
        if not path:
            return
        try:
            self.service.import_model(path)
        except Exception as exc:
            self._append_log(f"MODEL ERROR: {exc}")

    def _clear_model(self, *_args) -> None:
        try:
            self.service.clear_model()
        except Exception as exc:
            self._append_log(f"MODEL ERROR: {exc}")

    def _choose_save_folder(self, *_args) -> None:
        path = choose_directory(
            "Select packet archive folder", self.dpg.get_value("server.save_root")
        )
        if path:
            self.dpg.set_value("server.save_root", path)

    def _poll_events(self) -> None:
        while True:
            try:
                kind, payload = self.service.events.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                self._append_log(str(payload))
            elif kind == "server_status":
                running = payload == "running"
                self.dpg.set_value("server.status", "Running" if running else "Stopped")
                self.dpg.configure_item("server.status", color=GOOD if running else MUTED)
                self.dpg.configure_item("server.import_model", enabled=not running)
                self.dpg.configure_item(
                    "server.clear_model",
                    enabled=not running and self.service.model is not None,
                )
            elif kind == "model_loaded":
                self._show_model_info(payload)
            elif kind == "clients_changed":
                self._sync_subject_tabs()

    def _show_model_info(self, rows: list[tuple[str, str]]) -> None:
        values = dict(rows)
        loaded = bool(rows)
        self.dpg.set_value(
            "server.ml_summary",
            "Model loaded. Its exported ECG window, filter, feature extraction, and estimator pipeline will be used unchanged."
            if loaded
            else "No model loaded. Import a current model artifact before the server can start.",
        )
        if self.dpg.does_item_exist("server.start"):
            self.dpg.configure_item(
                "server.start", enabled=loaded and not self.service.running.is_set()
            )
            self.dpg.configure_item(
                "server.clear_model", enabled=loaded and not self.service.running.is_set()
            )
        self.dpg.set_value("server.ml_type", values.get("Type", "-"))
        self.dpg.set_value("server.ml_inputs", values.get("Inputs", "-"))
        self.dpg.set_value("server.ml_classes", values.get("Classes", "-"))
        self.dpg.set_value("server.ml_complexity", values.get("Complexity", "-"))
        self.dpg.set_value("server.ml_pipeline", values.get("Pipeline", "No pipeline available"))
        self.dpg.set_value("server.ml_artifact", values.get("Artifact", "No artifact selected"))
        self.dpg.set_value("server.ml_feature_names", values.get("Input names", "No inputs available"))
        parameter_rows = [
            (key.removeprefix("parameter."), value)
            for key, value in rows
            if key.startswith("parameter.")
        ]
        metadata_rows = [
            (key.removeprefix("metadata."), value)
            for key, value in rows
            if key.startswith("metadata.")
        ]
        replace_table_rows(self.dpg, "server.ml_parameters", parameter_rows)
        replace_table_rows(self.dpg, "server.ml_metadata", metadata_rows)

    def _sync_subject_tabs(self) -> None:
        connected = set(self.service.subject_ids())
        existing = set(self.subject_views)
        for subject_id in sorted(connected - existing):
            self._create_subject_tab(subject_id)
        for subject_id in sorted(existing - connected):
            view = self.subject_views.pop(subject_id)
            if self.dpg.does_item_exist(view["tab"]):
                self.dpg.delete_item(view["tab"])
            if self.dpg.does_item_exist(view.get("texture", "")):
                self.dpg.delete_item(view["texture"])

    def _refresh_tables(self) -> None:
        connections = self.service.connection_rows()
        replace_table_rows(
            self.dpg,
            "server.connections_table",
            (
                (
                    row["id"],
                    row["length"],
                    row["ip"],
                    row["packages"],
                    row["ping"],
                    row["jitter"],
                    row["processing_latency"],
                    row["quality"],
                )
                for row in connections
            ),
        )
        persons = self.service.person_rows()
        replace_table_rows(
            self.dpg,
            "server.persons_table",
            (
                (
                    *_metadata_row(row["metadata"]),
                    row["signal_time"],
                    row["pulse"],
                )
                for row in persons
            ),
        )

    def _refresh_subject_view(self, subject_id: str) -> None:
        view = self.subject_views.get(subject_id)
        snapshot = self.service.subject_snapshot(subject_id)
        if not view or not snapshot:
            return
        dpg = self.dpg
        prefix = view["prefix"]
        assessment = snapshot["assessment"]
        if snapshot["bpm"] is None:
            pulse_text = "Pulse: n/a"
        else:
            pulse_text = f"Pulse: {snapshot['bpm']:.1f} bpm"
            if assessment and assessment.pulse_latest_rr_ms:
                pulse_text += (
                    f" | latest RR {assessment.pulse_latest_rr_ms:.0f} ms"
                    f" | {assessment.pulse_interval_count} recent intervals"
                )
        dpg.set_value(f"{prefix}.pulse", pulse_text)
        dpg.set_value(
            f"{prefix}.time", f"Recording time: {_duration(snapshot['signal_time'])}"
        )
        latency = snapshot.get("processing_latency_ms")
        dpg.set_value(
            f"{prefix}.latency",
            "Processing latency: n/a"
            if latency is None
            else f"Processing latency: {float(latency):.1f} ms",
        )
        if assessment:
            color = (
                MUTED
                if assessment.quality_class == 0
                else GOOD
                if assessment.quality_class == 1
                else WARNING
                if assessment.quality_class == 2
                else DANGER
            )
            dpg.set_value(
                f"{prefix}.quality",
                f"Quality: class {assessment.quality_class} - {assessment.quality_label} "
                f"({assessment.quality_score:.2f}, {assessment.source})",
            )
            dpg.configure_item(f"{prefix}.quality", color=color)
        duration = float(snapshot["history_duration"])
        dpg.configure_item(
            f"{prefix}.timeline", max_value=max(1.0, duration)
        )
        follow = bool(dpg.get_value(f"{prefix}.follow"))
        if follow:
            end_seconds = duration
            dpg.set_value(f"{prefix}.timeline", duration)
        else:
            end_seconds = min(duration, float(dpg.get_value(f"{prefix}.timeline")))
        window_seconds = float(dpg.get_value(f"{prefix}.window"))
        signal_window = self.service.subject_window(
            subject_id, end_seconds, window_seconds
        )
        if signal_window is None:
            return
        self._update_ecg_plot(subject_id, signal_window, window_seconds)
        self._update_acc_plot(subject_id, signal_window, window_seconds)
        self._update_subject_confusion(subject_id, snapshot)

    def _update_subject_confusion(
        self, subject_id: str, snapshot: dict[str, Any]
    ) -> None:
        view = self.subject_views[subject_id]
        labelled = int(snapshot.get("labelled_predictions", 0))
        bucket = labelled // 5
        if labelled == 0 or bucket == view["confusion_refresh_bucket"]:
            return
        view["confusion_refresh_bucket"] = bucket
        matrix = np.asarray(snapshot["confusion_matrix"], dtype=int)
        classes = [int(value) for value in snapshot["confusion_classes"]]
        rgba = render_confusion_matrix_array_rgba(
            matrix,
            classes,
            f"Subject {subject_id} confusion matrix",
            620,
            540,
        )
        self.dpg.set_value(
            view["texture"],
            (rgba.astype(np.float32) / 255.0).reshape(-1).tolist(),
        )
        metrics, summary = _confusion_metrics(matrix, classes)
        self.dpg.set_value(f"{view['prefix']}.confusion_summary", summary)
        replace_table_rows(
            self.dpg, f"{view['prefix']}.confusion_metrics", metrics
        )

    def _update_ecg_plot(
        self, subject_id: str, signal_window: dict[str, Any], window_seconds: float
    ) -> None:
        dpg = self.dpg
        prefix = self.subject_views[subject_id]["prefix"]
        unit = signal_window["ecg_units"][0] if signal_window["ecg_units"] else "mV"
        raw = to_millivolts(np.asarray(signal_window["raw_ecg"])[:, 0], unit)
        filtered = to_millivolts(
            np.asarray(signal_window["filtered_ecg"])[:, 0], unit
        )
        x_raw, y_raw = timed_axis(
            raw,
            signal_window["ecg_fs"],
            start_seconds=signal_window["start_seconds"],
        )
        x_filtered, y_filtered = timed_axis(
            filtered,
            signal_window["ecg_fs"],
            start_seconds=signal_window["start_seconds"],
        )
        dpg.set_value(f"{prefix}.raw", [x_raw, y_raw])
        dpg.set_value(f"{prefix}.filtered", [x_filtered, y_filtered])
        mode = dpg.get_value(f"{prefix}.signal_mode")
        selected = raw if mode == "Raw" else filtered
        if mode == "Both":
            selected = np.concatenate((raw, filtered))
        center = float(np.median(selected)) if len(selected) else 0.0
        axis_end = max(window_seconds, float(signal_window["end_seconds"]))
        axis_start = max(0.0, axis_end - window_seconds)
        speed = float(dpg.get_value(f"{prefix}.speed"))
        gain = float(dpg.get_value(f"{prefix}.gain"))
        size = dpg.get_item_rect_size(f"{prefix}.ecg_plot")
        width = max(400.0, float(size[0] if size else 1000.0))
        height = max(250.0, float(size[1] if size else 430.0))
        y_span = max(1.0, height * window_seconds * speed / (width * gain))
        y_low, y_high = center - y_span / 2.0, center + y_span / 2.0
        dpg.set_axis_limits(f"{prefix}.ecg_x", axis_start, axis_end)
        dpg.set_axis_limits(f"{prefix}.ecg_y", y_low, y_high)
        minor_x, major_x = 1.0 / speed, 5.0 / speed
        minor_y, major_y = 1.0 / gain, 5.0 / gain
        dpg.set_value(
            f"{prefix}.grid_v_minor",
            [_grid_values(axis_start, axis_end, minor_x, major_x)],
        )
        dpg.set_value(
            f"{prefix}.grid_v_major", [_aligned_values(axis_start, axis_end, major_x)]
        )
        dpg.set_value(
            f"{prefix}.grid_h_minor", [_grid_values(y_low, y_high, minor_y, major_y)]
        )
        dpg.set_value(
            f"{prefix}.grid_h_major", [_aligned_values(y_low, y_high, major_y)]
        )
        dpg.set_value(
            f"{prefix}.calibration",
            f"{speed:g} mm/s | {gain:g} mm/mV | small box "
            f"{minor_x:.3f} s x {minor_y:.2f} mV | large box "
            f"{major_x:.2f} s x {major_y:.2f} mV",
        )

    def _update_acc_plot(
        self, subject_id: str, signal_window: dict[str, Any], window_seconds: float
    ) -> None:
        acceleration = signal_window["acceleration"]
        if acceleration is None:
            return
        dpg = self.dpg
        prefix = self.subject_views[subject_id]["prefix"]
        values = np.asarray(acceleration)
        for index, tag in enumerate(("acc_x", "acc_y", "acc_z")):
            if index < values.shape[1]:
                x, y = timed_axis(
                    values[:, index],
                    signal_window["acc_fs"],
                    start_seconds=signal_window["start_seconds"],
                )
                dpg.set_value(f"{prefix}.{tag}", [x, y])
        axis_end = max(window_seconds, float(signal_window["end_seconds"]))
        axis_start = max(0.0, axis_end - window_seconds)
        dpg.set_axis_limits(f"{prefix}.acc_axis_x", axis_start, axis_end)
        dpg.fit_axis_data(f"{prefix}.acc_axis_y")

    def _load_subject_settings(self, subject_id: str, snapshot: dict[str, Any]) -> None:
        prefix = self.subject_views[subject_id]["prefix"]
        config = snapshot["filter_config"]
        for suffix, value in (
            ("filter_dc", config.remove_dc),
            ("filter_kind", config.filter_kind),
            ("filter_order", config.order),
            ("filter_low", config.low_hz),
            ("filter_high", config.high_hz),
            ("filter_ripple", config.ripple_db),
        ):
            self.dpg.set_value(f"{prefix}.{suffix}", value)

    def _render_subject_metadata(
        self, subject_id: str, metadata: dict[str, Any]
    ) -> None:
        view = self.subject_views.get(subject_id)
        if not view:
            return
        replace_table_rows(
            self.dpg,
            f"{view['prefix']}.metadata",
            [_metadata_row(metadata)],
        )

    def _alert_clicked(self, _sender, _app_data, subject_id) -> None:
        self._send_alert(str(subject_id))

    def _signal_mode_changed(self, _sender, _app_data, subject_id) -> None:
        subject_id = str(subject_id)
        self._set_signal_visibility(subject_id)
        self._refresh_subject_view(subject_id)

    def _subject_display_changed(self, _sender, _app_data, subject_id) -> None:
        self._refresh_subject_view(str(subject_id))

    def _timeline_changed_callback(self, _sender, _app_data, subject_id) -> None:
        subject_id = str(subject_id)
        self._timeline_changed(subject_id)
        self._refresh_subject_view(subject_id)

    def _jump_latest_clicked(self, _sender, _app_data, subject_id) -> None:
        subject_id = str(subject_id)
        self._jump_latest(subject_id)
        self._refresh_subject_view(subject_id)

    def _filter_clicked(self, _sender, _app_data, subject_id) -> None:
        self._apply_filter(str(subject_id))

    def _set_signal_visibility(self, subject_id: str) -> None:
        prefix = self.subject_views[subject_id]["prefix"]
        mode = self.dpg.get_value(f"{prefix}.signal_mode")
        self.dpg.configure_item(f"{prefix}.raw", show=mode in {"Raw", "Both"})
        self.dpg.configure_item(
            f"{prefix}.filtered", show=mode in {"Filtered", "Both"}
        )

    def _timeline_changed(self, subject_id: str) -> None:
        view = self.subject_views.get(subject_id)
        if view:
            self.dpg.set_value(f"{view['prefix']}.follow", False)

    def _jump_latest(self, subject_id: str) -> None:
        view = self.subject_views.get(subject_id)
        if view:
            self.dpg.set_value(f"{view['prefix']}.follow", True)

    def _apply_filter(self, subject_id: str) -> None:
        view = self.subject_views.get(subject_id)
        if not view:
            return
        prefix = view["prefix"]
        try:
            config = FilterConfig(
                remove_dc=bool(self.dpg.get_value(f"{prefix}.filter_dc")),
                filter_kind=self.dpg.get_value(f"{prefix}.filter_kind"),
                order=safe_int(self.dpg.get_value(f"{prefix}.filter_order"), "Filter order"),
                low_hz=safe_float(self.dpg.get_value(f"{prefix}.filter_low"), "Low frequency"),
                high_hz=safe_float(self.dpg.get_value(f"{prefix}.filter_high"), "High frequency"),
                ripple_db=safe_float(self.dpg.get_value(f"{prefix}.filter_ripple"), "Ripple"),
            )
            self.service.set_filter(subject_id, config)
            self.dpg.set_value(
                f"{prefix}.filter_message",
                "Display/pulse filter applied. Model preprocessing remains unchanged.",
            )
            self.dpg.configure_item(f"{prefix}.filter_message", color=GOOD)
            self._refresh_subject_view(subject_id)
        except Exception as exc:
            self.dpg.set_value(f"{prefix}.filter_message", str(exc))
            self.dpg.configure_item(f"{prefix}.filter_message", color=DANGER)

    def _send_alert(self, subject_id: str) -> None:
        prefix = self.subject_views[subject_id]["prefix"]
        try:
            self.service.send_medical_alert(
                subject_id, self.dpg.get_value(f"{prefix}.alert_text")
            )
            self.dpg.set_value(
                f"{prefix}.alert_status", "Medical alert delivered to the client."
            )
            self.dpg.configure_item(f"{prefix}.alert_status", color=GOOD)
        except Exception as exc:
            self.dpg.set_value(f"{prefix}.alert_status", f"Alert failed: {exc}")
            self.dpg.configure_item(f"{prefix}.alert_status", color=DANGER)
            self._append_log(f"ALERT ERROR for {subject_id}: {exc}")

    def _append_log(self, message: str) -> None:
        from datetime import datetime

        self.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        self.logs = self.logs[-250:]
        if self.dpg.does_item_exist("server.log"):
            self.dpg.set_value("server.log", "\n".join(self.logs))


def _aligned_values(start: float, end: float, step: float) -> list[float]:
    if step <= 0 or end <= start:
        return []
    first = math.ceil((start - 1e-9) / step) * step
    count = max(0, int(math.floor((end - first + 1e-9) / step)) + 1)
    return [first + index * step for index in range(min(count, 5000))]


def _metadata_row(metadata: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        _format_metadata_value(path, _nested_metadata_value(metadata, path))
        for path, _label, _width in SUBJECT_METADATA_COLUMNS
    )


def _nested_metadata_value(metadata: dict[str, Any], path: str) -> Any:
    value: Any = metadata
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _format_metadata_value(path: str, value: Any) -> str:
    if value is None or value == "":
        return "-"
    if path == "smoker":
        return "Yes" if str(value).strip().lower() in {"1", "true", "yes"} else "No"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value) if value else "-"
    if isinstance(value, dict):
        return ", ".join(f"{key}: {item}" for key, item in value.items())
    if path.endswith("sample_count"):
        try:
            return f"{int(value):,}"
        except (TypeError, ValueError):
            pass
    if path.endswith(".fs"):
        try:
            return f"{float(value):g}"
        except (TypeError, ValueError):
            pass
    return str(value)


def _grid_values(
    start: float, end: float, minor_step: float, major_step: float
) -> list[float]:
    values = _aligned_values(start, end, minor_step)
    return [
        value
        for value in values
        if abs(value / major_step - round(value / major_step)) > 1e-6
    ]


def _duration(seconds: float) -> str:
    value = max(0, int(seconds))
    hours, remainder = divmod(value, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _confusion_metrics(
    matrix: np.ndarray, classes: list[int]
) -> tuple[list[tuple[Any, ...]], str]:
    matrix = np.asarray(matrix, dtype=float)
    supports = matrix.sum(axis=1)
    predicted_counts = matrix.sum(axis=0)
    recalls: list[float] = []
    f1_values: list[float] = []
    rows: list[tuple[Any, ...]] = []
    for index, quality_class in enumerate(classes):
        true_positive = float(matrix[index, index])
        precision = true_positive / predicted_counts[index] if predicted_counts[index] else 0.0
        recall = true_positive / supports[index] if supports[index] else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1_values.append(f1)
        rows.append(
            (
                quality_class,
                int(supports[index]),
                int(predicted_counts[index]),
                f"{precision:.3f}",
                f"{recall:.3f}",
                f"{f1:.3f}",
            )
        )
    total = float(matrix.sum())
    accuracy = float(np.trace(matrix) / total) if total else 0.0
    balanced = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1_values)) if f1_values else 0.0
    dangerous = 0
    if 3 in classes and 1 in classes:
        dangerous = int(matrix[classes.index(3), classes.index(1)])
    summary = (
        f"Labelled windows: {int(total):,} | Accuracy: {accuracy:.3f} | "
        f"Balanced accuracy: {balanced:.3f} | Macro F1: {macro_f1:.3f} | "
        f"Dangerous Class 3 -> Class 1 errors: {dangerous:,}"
    )
    return rows, summary


def main() -> None:
    ServerApp().run()


if __name__ == "__main__":
    main()
