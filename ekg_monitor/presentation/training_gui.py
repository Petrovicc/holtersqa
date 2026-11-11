from __future__ import annotations

import json
import math
import queue
import threading
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ekg_monitor.application.models import (
    FILTER_KINDS,
    DatasetBuildConfig,
    FeatureConfig,
    FilterConfig,
    TrainingConfig,
)
from ekg_monitor.application.optimization_service import (
    OptimizationConfig,
    OptimizationRun,
    optimization_size,
    parse_mlp_architectures,
    parse_window_lengths,
    run_optimization,
)
from ekg_monitor.application.training_service import (
    TrainingResult,
    build_dataset,
    dataset_summary,
    load_dataframe,
    save_dataframe,
    save_training_report,
    train_model,
)
from ekg_monitor.infrastructure.butqdb_repository import ButqdbRepository
from ekg_monitor.infrastructure.model_repository import save_model_bundle
from ekg_monitor.infrastructure.training_report_exporter import (
    export_training_artifacts,
    render_confusion_matrix_rgba,
)

from .common import ACCENT, DANGER, GOOD, MUTED, WARNING, configure_global_theme, replace_table_rows, safe_float, safe_int
from .native_dialogs import choose_directory, choose_open_file, choose_save_file


class TrainingApp:
    def __init__(self) -> None:
        import dearpygui.dearpygui as dpg

        self.dpg = dpg
        self.repository = ButqdbRepository()
        self.frame: pd.DataFrame | None = None
        self.result: TrainingResult | None = None
        self.build_config: DatasetBuildConfig | None = None
        self.optimization_run: OptimizationRun | None = None
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.cancel = threading.Event()
        self.worker: threading.Thread | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        dpg = self.dpg
        dpg.create_context()
        with dpg.texture_registry():
            dpg.add_dynamic_texture(
                720,
                620,
                np.zeros((620, 720, 4), dtype=np.float32).reshape(-1).tolist(),
                tag="training.confusion_texture",
            )
        self.theme = configure_global_theme(dpg)
        with dpg.window(tag="training.main", label="HolterSQA Model Training"):
            with dpg.group(horizontal=True):
                dpg.add_text("HolterSQA Model Training", tag="training.heading", color=ACCENT)
                dpg.add_text("Idle", tag="training.status", color=MUTED)
                dpg.add_button(
                    tag="training.theme_toggle",
                    label="Light mode",
                    callback=lambda: self.theme.toggle("training.theme_toggle"),
                )
            dpg.add_progress_bar(tag="training.progress", default_value=0.0, overlay="0%", width=-1)
            with dpg.tab_bar():
                with dpg.tab(label="1. Prepare Dataset"):
                    self._dataset_tab()
                with dpg.tab(label="2. Train Models"):
                    self._training_tab()
                with dpg.tab(label="3. Optimize Solutions"):
                    self._optimization_tab()
            dpg.add_input_text(tag="training.log", label="Activity", multiline=True, readonly=True, height=100, width=-1)
        self.theme.bind_heading("training.heading")
        self.theme.bind_role("training.loss_series", "loss")
        self.theme.bind_role("training.accuracy_series", "accuracy")
        dpg.create_viewport(title="HolterSQA Model Training", width=1450, height=920, min_width=1080, min_height=720)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("training.main", True)
        dpg.set_exit_callback(self._cancel)
        default_root = Path.cwd() / "butqdb"
        if default_root.is_dir():
            dpg.set_value("training.dataset_root", str(default_root))
            self._scan()

    def _dataset_tab(self) -> None:
        dpg = self.dpg
        with dpg.group(horizontal=True):
            with dpg.child_window(width=510, height=-1, border=True):
                dpg.add_text("SOURCE", color=ACCENT)
                dpg.add_input_text(tag="training.dataset_root", label="BUT QDB folder", width=-120)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Browse", callback=self._choose_root)
                    dpg.add_button(label="Scan", callback=self._scan)
                dpg.add_input_text(tag="training.records", label="Record IDs", multiline=True, height=75, hint="Comma-separated; blank means all scanned records")
                dpg.add_text("No records scanned", tag="training.scan_summary", color=MUTED)
                dpg.add_separator()
                dpg.add_text("WINDOWS AND LABELS", color=ACCENT)
                dpg.add_input_float(tag="training.window", label="Window seconds", default_value=10.0)
                dpg.add_slider_float(tag="training.overlap", label="Overlap", default_value=0.5, min_value=0.0, max_value=0.95, format="%.2f")
                dpg.add_combo(
                    ["consensus", "majority", "annotator1", "annotator2", "annotator3"],
                    tag="training.labels",
                    label="Label source",
                    default_value="consensus",
                )
                dpg.add_input_int(
                    tag="training.max_windows",
                    label="Max windows / record (0 = all)",
                    default_value=0,
                    min_value=0,
                )
                dpg.add_combo(["none", "undersample"], tag="training.balance", label="Class balancing", default_value="none")
                dpg.add_separator()
                dpg.add_text("FILTER", color=ACCENT)
                dpg.add_checkbox(tag="training.filter_dc", label="Remove DC", default_value=True)
                dpg.add_combo(list(FILTER_KINDS), tag="training.filter_kind", label="Type", default_value="butter_bandpass")
                dpg.add_input_int(tag="training.filter_order", label="Order / taps", default_value=4)
                dpg.add_input_float(tag="training.filter_low", label="Low Hz", default_value=0.5)
                dpg.add_input_float(tag="training.filter_high", label="High Hz", default_value=40.0)
                dpg.add_input_float(tag="training.filter_ripple", label="Ripple dB", default_value=0.5)
            with dpg.child_window(width=-1, height=-1, border=True):
                dpg.add_text("FEATURES", color=ACCENT)
                with dpg.group(horizontal=True):
                    dpg.add_checkbox(tag="training.feat_ecg_time", label="ECG time statistics", default_value=True)
                    dpg.add_checkbox(tag="training.feat_ecg_spectrum", label="ECG spectrum", default_value=True)
                    dpg.add_checkbox(tag="training.feat_acc", label="Accelerometer", default_value=True)
                    dpg.add_checkbox(tag="training.feat_acc_diff", label="Acceleration difference", default_value=True)
                    dpg.add_checkbox(tag="training.feat_demo", label="Demographics", default_value=True)
                dpg.add_input_int(tag="training.spectrum_bins", label="Spectrum bins", default_value=32, min_value=4, max_value=256)
                with dpg.group(horizontal=True):
                    dpg.add_button(tag="training.build", label="Build prepared dataset", callback=self._build_dataset)
                    dpg.add_button(label="Load prepared dataset", callback=self._choose_load)
                    dpg.add_button(label="Save prepared dataset", callback=self._choose_save)
                    dpg.add_button(tag="training.cancel", label="Cancel", callback=self._cancel, enabled=False)
                dpg.add_text("No prepared dataset", tag="training.dataset_summary", wrap=780)
                with dpg.table(
                    tag="training.class_table",
                    header_row=True,
                    row_background=True,
                    borders_innerH=True,
                    borders_outerH=True,
                    borders_innerV=True,
                    borders_outerV=True,
                    height=180,
                ):
                    dpg.add_table_column(label="Quality class")
                    dpg.add_table_column(label="Window count")
                dpg.add_text(
                    "Quality classes: 1 = diagnostically good; 2 = QRS usable but detailed waveforms unreliable; 3 = unusable.",
                    color=MUTED,
                    wrap=760,
                )

    def _training_tab(self) -> None:
        dpg = self.dpg
        with dpg.group(horizontal=True):
            with dpg.child_window(width=485, height=-1, border=True):
                dpg.add_text("MODEL AND VALIDATION", tag="training.model_heading", color=ACCENT)
                self.theme.bind_heading("training.model_heading")
                dpg.add_combo(
                    ["MLP", "Logistic Regression", "Random Forest", "Gradient Boosting"],
                    tag="training.model_type",
                    label="Model",
                    default_value="MLP",
                    callback=self._model_changed,
                )
                dpg.add_slider_float(tag="training.validation_fraction", label="Validation fraction", default_value=0.15, min_value=0.05, max_value=0.4, format="%.2f")
                dpg.add_slider_float(tag="training.final_test_fraction", label="Final-test fraction", default_value=0.15, min_value=0.05, max_value=0.4, format="%.2f")
                dpg.add_input_int(tag="training.seed", label="Random seed", default_value=42)
                dpg.add_checkbox(tag="training.standardize", label="Standardize features", default_value=True)
                dpg.add_checkbox(tag="training.subject_split", label="Split by person (prevents subject leakage)", default_value=True)
                dpg.add_combo(
                    ["Stratified holdout", "Stratified Group K-Fold combinations"],
                    tag="training.split_strategy",
                    label="Evaluation design",
                    default_value="Stratified holdout",
                )
                dpg.add_input_int(tag="training.group_folds", label="Group folds", default_value=5, min_value=3, max_value=10)
                dpg.add_combo(
                    ["oversample", "undersample", "none"],
                    tag="training.class_equalization",
                    label="Equalize training classes",
                    default_value="oversample",
                )
                dpg.add_text(
                    "Equalization is applied independently after splitting; samples never cross between training and validation.",
                    color=MUTED,
                    wrap=430,
                )
                dpg.add_checkbox(
                    tag="training.equalize_validation",
                    label="Equalize validation classes too",
                    default_value=True,
                )
                dpg.add_checkbox(
                    tag="training.class_weight",
                    label="Balanced model weights when equalization is none",
                    default_value=True,
                )
                dpg.add_separator()
                dpg.add_text("MODEL SELECTION TARGET", color=ACCENT)
                dpg.add_combo(
                    [
                        "Loss",
                        "Accuracy",
                        "Balanced Accuracy",
                        "Precision",
                        "Recall",
                        "F1",
                    ],
                    tag="training.selection_metric",
                    label="Validation target",
                    default_value="Accuracy",
                )
                dpg.add_combo(
                    ["Overall", "Class 1", "Class 2", "Class 3"],
                    tag="training.selection_scope",
                    label="Optimization scope",
                    default_value="Overall",
                )
                dpg.add_text(
                    "Overall precision/recall/F1 use macro averaging. Class accuracy is the correctly detected fraction of that actual class (the confusion-matrix row accuracy).",
                    color=MUTED,
                    wrap=430,
                )
                dpg.add_checkbox(
                    tag="training.error_penalty",
                    label="Penalize dangerous Class 3 -> Class 1 errors",
                    default_value=True,
                )
                dpg.add_input_float(
                    tag="training.error_penalty_weight",
                    label="Dangerous-error penalty",
                    default_value=5.0,
                    min_value=1.0,
                    format="%.1f",
                )
                dpg.add_input_int(
                    tag="training.minimum_samples",
                    label="Minimum samples / evaluation class",
                    default_value=100,
                    min_value=1,
                )
                dpg.add_separator()
                with dpg.group(tag="training.mlp_group"):
                    dpg.add_text("MLP PARAMETERS", color=ACCENT)
                    dpg.add_input_text(tag="training.mlp_layers", label="Hidden layers", default_value="128,64")
                    dpg.add_input_int(tag="training.mlp_epochs", label="Epochs", default_value=50, min_value=1)
                    dpg.add_input_float(tag="training.learning_rate", label="Learning rate", default_value=0.001, format="%.6f")
                    dpg.add_checkbox(
                        tag="training.early_stopping",
                        label="Early stopping",
                        default_value=True,
                    )
                    dpg.add_input_int(
                        tag="training.early_patience",
                        label="Patience (epochs)",
                        default_value=15,
                        min_value=1,
                    )
                    dpg.add_input_float(
                        tag="training.early_delta",
                        label="Minimum validation improvement",
                        default_value=0.001,
                        min_value=0.0,
                        format="%.4f",
                    )
                with dpg.group(tag="training.rf_group", show=False):
                    dpg.add_text("RANDOM FOREST PARAMETERS", color=ACCENT)
                    dpg.add_input_int(tag="training.rf_estimators", label="Trees", default_value=300, min_value=1)
                    dpg.add_input_int(tag="training.rf_depth", label="Maximum depth (0 = unlimited)", default_value=0, min_value=0)
                dpg.add_separator()
                with dpg.group(horizontal=True):
                    dpg.add_button(tag="training.train", label="Train model", callback=self._train)
                    dpg.add_button(
                        tag="training.export",
                        label="Export model and report",
                        callback=self._choose_export,
                        enabled=False,
                    )
                dpg.add_text(
                    "Only parameters relevant to the selected model are shown.",
                    color=MUTED,
                    wrap=430,
                )
            with dpg.child_window(width=-1, height=-1, border=True):
                dpg.add_text("TRAINING RESULTS", tag="training.results_heading", color=ACCENT)
                self.theme.bind_heading("training.results_heading")
                with dpg.table(header_row=False, policy=dpg.mvTable_SizingStretchProp):
                    for _ in range(4):
                        dpg.add_table_column()
                    with dpg.table_row():
                        for title, tag in (
                            ("ACCURACY", "training.metric_accuracy"),
                            ("MACRO F1", "training.metric_f1"),
                            ("TRAIN WINDOWS", "training.metric_train"),
                            ("FINAL TEST WINDOWS", "training.metric_test"),
                        ):
                            with dpg.table_cell():
                                with dpg.child_window(height=92, border=True):
                                    dpg.add_text(title, color=MUTED)
                                    dpg.add_text("-", tag=tag)
                                    self.theme.bind_heading(tag, metric=True)
                with dpg.collapsing_header(label="Detailed evaluation", default_open=True):
                    dpg.add_text("", tag="training.evaluation_warning", color=WARNING, wrap=1300)
                    with dpg.table(
                        header_row=False,
                        policy=dpg.mvTable_SizingStretchProp,
                        borders_innerV=False,
                    ):
                        for _ in range(3):
                            dpg.add_table_column()
                        with dpg.table_row():
                            with dpg.table_cell():
                                with dpg.child_window(height=245, border=True):
                                    dpg.add_text("OVERALL PERFORMANCE", color=ACCENT)
                                    with dpg.table(
                                        tag="training.overall_table",
                                        header_row=True,
                                        row_background=True,
                                        borders_innerH=True,
                                        borders_outerH=True,
                                    ):
                                        dpg.add_table_column(label="Metric")
                                        dpg.add_table_column(label="Value")
                            with dpg.table_cell():
                                with dpg.child_window(height=245, border=True):
                                    dpg.add_text("CLASS BALANCE", color=ACCENT)
                                    with dpg.table(
                                        tag="training.balance_table",
                                        header_row=True,
                                        row_background=True,
                                        borders_innerH=True,
                                        borders_outerH=True,
                                    ):
                                        for label in (
                                            "Class",
                                            "Train pre",
                                            "Train used",
                                            "Val pre",
                                            "Val used",
                                            "Final test",
                                        ):
                                            dpg.add_table_column(label=label)
                            with dpg.table_cell():
                                with dpg.child_window(height=245, border=True):
                                    dpg.add_text("PER-CLASS PERFORMANCE", color=ACCENT)
                                    with dpg.table(
                                        tag="training.classification_table",
                                        header_row=True,
                                        row_background=True,
                                        borders_innerH=True,
                                        borders_outerH=True,
                                    ):
                                        for label in (
                                            "Class",
                                            "Precision",
                                            "Recall",
                                            "F1",
                                            "Support",
                                            "95% CI",
                                        ):
                                            dpg.add_table_column(label=label)
                with dpg.collapsing_header(label="Subjects and data allocation", default_open=True):
                    dpg.add_text("", tag="training.subject_summary", wrap=1300)
                    with dpg.table(
                        tag="training.subject_allocation_table",
                        header_row=True,
                        row_background=True,
                        borders_innerH=True,
                        borders_outerH=True,
                        borders_innerV=True,
                        borders_outerV=True,
                        scrollY=True,
                        height=190,
                    ):
                        for label in (
                            "Subject",
                            "Total",
                            "Train count",
                            "Train %",
                            "Validation count",
                            "Validation %",
                            "Final test count",
                            "Final test %",
                        ):
                            dpg.add_table_column(label=label)
                with dpg.collapsing_header(
                    label="Stratified fold-combination results", default_open=False
                ):
                    with dpg.table(
                        tag="training.evaluation_candidates_table",
                        header_row=True,
                        row_background=True,
                        borders_innerH=True,
                        borders_outerH=True,
                        borders_innerV=True,
                        borders_outerV=True,
                        scrollX=True,
                        scrollY=True,
                        height=190,
                    ):
                        for label in (
                            "Candidate",
                            "Selected",
                            "Validation objective",
                            "Final accuracy",
                            "Final balanced accuracy",
                            "Final macro F1",
                            "Dangerous 3 -> 1",
                            "Train subjects",
                            "Validation subjects",
                            "Final-test subjects",
                        ):
                            dpg.add_table_column(label=label)
                with dpg.tab_bar():
                    with dpg.tab(label="Learning curves"):
                        with dpg.table(
                            header_row=False,
                            policy=dpg.mvTable_SizingStretchProp,
                            borders_innerV=False,
                        ):
                            dpg.add_table_column()
                            dpg.add_table_column()
                            with dpg.table_row():
                                with dpg.table_cell():
                                    with dpg.plot(label="Training loss", height=385, width=-1):
                                        dpg.add_plot_axis(dpg.mvXAxis, label="Epoch", tag="training.loss_x")
                                        with dpg.plot_axis(dpg.mvYAxis, label="Loss", tag="training.loss_y"):
                                            dpg.add_line_series([], [], label="Loss", tag="training.loss_series")
                                with dpg.table_cell():
                                    with dpg.plot(label="Validation objective", tag="training.objective_plot", height=385, width=-1):
                                        dpg.add_plot_axis(dpg.mvXAxis, label="Epoch", tag="training.accuracy_x")
                                        with dpg.plot_axis(dpg.mvYAxis, label="Objective score", tag="training.accuracy_y"):
                                            dpg.add_line_series([], [], label="Objective", tag="training.accuracy_series")
                    with dpg.tab(label="Confusion matrix"):
                        dpg.add_text(
                            "Rows are actual quality classes; columns are predicted quality classes. Percentages are normalized within each actual-class row.",
                            color=MUTED,
                        )
                        with dpg.group(horizontal=True):
                            dpg.add_spacer(width=90)
                            dpg.add_image(
                                "training.confusion_texture",
                                tag="training.confusion_image",
                                width=720,
                                height=620,
                            )

    def _optimization_tab(self) -> None:
        dpg = self.dpg
        with dpg.group(horizontal=True):
            with dpg.child_window(width=540, height=-1, border=True):
                dpg.add_text("SEARCH SPACE", color=ACCENT)
                dpg.add_input_text(
                    tag="optimization.run_name",
                    label="Run name (optional)",
                    hint="Generated automatically when blank",
                )
                dpg.add_input_text(
                    tag="optimization.windows",
                    label="Window lengths (seconds)",
                    default_value="3,5,10",
                )
                dpg.add_text(
                    "Every non-empty combination of the selected feature groups will be prepared for every window length.",
                    color=MUTED,
                    wrap=490,
                )
                with dpg.group(horizontal=True):
                    dpg.add_checkbox(
                        tag="optimization.feat_ecg_time",
                        label="ECG time",
                        default_value=True,
                    )
                    dpg.add_checkbox(
                        tag="optimization.feat_ecg_spectrum",
                        label="ECG spectrum",
                        default_value=True,
                    )
                    dpg.add_checkbox(
                        tag="optimization.feat_acc",
                        label="ACC",
                        default_value=True,
                    )
                with dpg.group(horizontal=True):
                    dpg.add_checkbox(
                        tag="optimization.feat_acc_diff",
                        label="ACC difference",
                        default_value=True,
                    )
                    dpg.add_checkbox(
                        tag="optimization.feat_demo",
                        label="Demographics",
                        default_value=False,
                    )
                dpg.add_separator()
                dpg.add_text("MODEL FAMILIES", color=ACCENT)
                with dpg.group(horizontal=True):
                    dpg.add_checkbox(
                        tag="optimization.model_mlp", label="MLP", default_value=True
                    )
                    dpg.add_checkbox(
                        tag="optimization.model_logistic",
                        label="Logistic regression",
                        default_value=True,
                    )
                with dpg.group(horizontal=True):
                    dpg.add_checkbox(
                        tag="optimization.model_rf",
                        label="Random forest",
                        default_value=True,
                    )
                    dpg.add_checkbox(
                        tag="optimization.model_gb",
                        label="Gradient boosting",
                        default_value=True,
                    )
                dpg.add_input_text(
                    tag="optimization.architectures",
                    label="MLP hidden layers",
                    default_value="128,64; 64,128,64; 64,128,128,64",
                    multiline=True,
                    height=72,
                )
                dpg.add_input_int(
                    tag="optimization.epochs",
                    label="Maximum epochs",
                    default_value=200,
                    min_value=1,
                )
                dpg.add_input_int(
                    tag="optimization.patience",
                    label="Early-stopping patience",
                    default_value=15,
                    min_value=1,
                )
                dpg.add_input_float(
                    tag="optimization.min_delta",
                    label="Minimum validation improvement",
                    default_value=0.001,
                    min_value=0.0,
                    format="%.4f",
                )
                dpg.add_combo(
                    ["oversample", "undersample", "none"],
                    tag="optimization.equalization",
                    label="Class equalization",
                    default_value="oversample",
                )
                dpg.add_checkbox(
                    tag="optimization.standardize",
                    label="Standardize features",
                    default_value=True,
                )
                dpg.add_checkbox(
                    tag="optimization.subject_split",
                    label="Split by person",
                    default_value=True,
                )
                dpg.add_checkbox(
                    tag="optimization.equalize_validation",
                    label="Equalize validation independently",
                    default_value=True,
                )
                dpg.add_slider_float(
                    tag="optimization.validation_fraction",
                    label="Validation fraction",
                    default_value=0.15,
                    min_value=0.05,
                    max_value=0.4,
                    format="%.2f",
                )
                dpg.add_slider_float(
                    tag="optimization.final_test_fraction",
                    label="Final-test fraction",
                    default_value=0.15,
                    min_value=0.05,
                    max_value=0.4,
                    format="%.2f",
                )
                dpg.add_combo(
                    ["Stratified holdout", "Stratified Group K-Fold combinations"],
                    tag="optimization.split_strategy",
                    label="Evaluation design",
                    default_value="Stratified holdout",
                )
                dpg.add_input_int(
                    tag="optimization.group_folds",
                    label="Group folds",
                    default_value=5,
                    min_value=3,
                    max_value=10,
                )
                dpg.add_combo(
                    [
                        "Loss",
                        "Accuracy",
                        "Balanced Accuracy",
                        "Precision",
                        "Recall",
                        "F1",
                    ],
                    tag="optimization.selection_metric",
                    label="Validation target",
                    default_value="Accuracy",
                )
                dpg.add_combo(
                    ["Overall", "Class 1", "Class 2", "Class 3"],
                    tag="optimization.selection_scope",
                    label="Optimization scope",
                    default_value="Overall",
                )
                dpg.add_checkbox(
                    tag="optimization.error_penalty",
                    label="Penalize dangerous Class 3 -> Class 1 errors",
                    default_value=True,
                )
                dpg.add_input_float(
                    tag="optimization.error_penalty_weight",
                    label="Dangerous-error penalty",
                    default_value=5.0,
                    min_value=1.0,
                    format="%.1f",
                )
                dpg.add_input_int(
                    tag="optimization.minimum_samples",
                    label="Minimum samples / evaluation class",
                    default_value=100,
                    min_value=1,
                )
                dpg.add_separator()
                dpg.add_text("CPU AND MEMORY", color=ACCENT)
                dpg.add_input_int(
                    tag="optimization.max_workers",
                    label="Parallel CPU candidates (0 = automatic)",
                    default_value=0,
                    min_value=0,
                )
                dpg.add_slider_int(
                    tag="optimization.ram_fraction",
                    label="Available RAM budget",
                    default_value=75,
                    min_value=25,
                    max_value=90,
                    format="%d%%",
                )
                dpg.add_text(
                    "Automatic mode sizes parallel CPU processes from logical cores and available RAM. Search candidates use the same scikit-learn implementations as Train Models; CUDA is intentionally disabled for algorithm parity.",
                    tag="optimization.resource_note",
                    color=MUTED,
                    wrap=490,
                )
                dpg.add_separator()
                dpg.add_text("FIXED PREPARATION", color=ACCENT)
                dpg.add_text(
                    "Filter: none | DC removal: on. Overlap, labels, record selection, spectrum bins, and optional per-record cap are taken from Prepare Dataset.",
                    color=MUTED,
                    wrap=490,
                )
                dpg.add_input_text(
                    tag="optimization.database_root",
                    label="Dataset output",
                    default_value=str((Path.cwd() / "DBs").resolve()),
                )
                dpg.add_input_text(
                    tag="optimization.model_root",
                    label="Model output",
                    default_value=str((Path.cwd() / "Models").resolve()),
                )
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="Preview combinations",
                        callback=self._preview_optimization,
                    )
                    dpg.add_button(
                        tag="optimization.start",
                        label="Start optimization",
                        callback=self._start_optimization,
                    )
                    dpg.add_button(
                        tag="optimization.cancel",
                        label="Cancel",
                        callback=self._cancel,
                        enabled=False,
                    )
                dpg.add_text(
                    "Press Preview combinations to calculate the search size.",
                    tag="optimization.summary",
                    color=MUTED,
                    wrap=490,
                )
            with dpg.child_window(width=-1, height=-1, border=True):
                dpg.add_text("OPTIMIZATION RESULTS", color=ACCENT)
                dpg.add_text(
                    "Rows selected as best for accuracy, balanced accuracy, macro F1, speed, or artifact size are highlighted.",
                    color=MUTED,
                    wrap=900,
                )
                with dpg.table(
                    tag="optimization.results",
                    header_row=True,
                    row_background=True,
                    borders_innerH=True,
                    borders_outerH=True,
                    borders_innerV=True,
                    borders_outerV=True,
                    scrollX=True,
                    scrollY=True,
                    freeze_rows=1,
                    height=-1,
                    policy=dpg.mvTable_SizingFixedFit,
                ):
                    for label, width in (
                        ("Best for", 190),
                        ("Status", 80),
                        ("Window", 75),
                        ("Features", 220),
                        ("Model", 150),
                        ("Backend", 190),
                        ("Architecture", 130),
                        ("Accuracy", 85),
                        ("Balanced", 85),
                        ("Macro F1", 85),
                        ("Best epoch", 90),
                        ("Epochs", 75),
                        ("Seconds", 80),
                        ("Candidate / error", 330),
                    ):
                        dpg.add_table_column(label=label, width_fixed=True, init_width_or_weight=width)

    def run(self) -> None:
        dpg = self.dpg
        while dpg.is_dearpygui_running():
            self._poll_events()
            dpg.render_dearpygui_frame()
        self.cancel.set()
        dpg.destroy_context()

    def _choose_root(self, *_args) -> None:
        path = choose_directory(
            "Select BUT QDB folder", self.dpg.get_value("training.dataset_root")
        )
        if path:
            self.dpg.set_value("training.dataset_root", path)
            self._scan()

    def _choose_load(self, *_args) -> None:
        path = choose_open_file(
            "Load prepared ECG dataset",
            (("Prepared dataset", "*.parquet *.csv"), ("All files", "*.*")),
            Path.cwd(),
        )
        if path:
            self._load_selected(None, {"file_path_name": path})

    def _choose_save(self, *_args) -> None:
        path = choose_save_file(
            "Save prepared ECG dataset",
            (("Parquet", "*.parquet"), ("CSV", "*.csv")),
            ".parquet",
            Path.cwd(),
            "training_dataset.parquet",
        )
        if path:
            self._save_selected(None, {"file_path_name": path})

    def _choose_export(self, *_args) -> None:
        directory = choose_directory("Export trained model and evaluation report", Path.cwd())
        if directory:
            self._export_selected(None, {"file_path_name": directory})

    def _model_changed(self, *_args) -> None:
        model_type = self.dpg.get_value("training.model_type")
        self.dpg.configure_item("training.mlp_group", show=model_type == "MLP")
        self.dpg.configure_item(
            "training.rf_group", show=model_type == "Random Forest"
        )

    def _scan(self, *_args) -> None:
        try:
            records = self.repository.scan(self.dpg.get_value("training.dataset_root"))
            ids = [record.record_id for record in records if record.ecg_header and record.annotation_path]
            self.dpg.set_value("training.records", ", ".join(ids))
            self.dpg.set_value("training.scan_summary", f"{len(ids)} annotated ECG records are available")
            self._log(f"Scanned {len(ids)} annotated records")
        except Exception as exc:
            self._error(str(exc))

    def _selected_records(self) -> list[str]:
        value = self.dpg.get_value("training.records")
        selected = [part.strip() for part in str(value).replace("\n", ",").split(",") if part.strip()]
        return selected or list(self.repository.records)

    def _dataset_config(self) -> DatasetBuildConfig:
        dpg = self.dpg
        filter_config = FilterConfig(
            remove_dc=bool(dpg.get_value("training.filter_dc")),
            filter_kind=dpg.get_value("training.filter_kind"),
            order=safe_int(dpg.get_value("training.filter_order"), "Filter order"),
            low_hz=safe_float(dpg.get_value("training.filter_low"), "Low frequency"),
            high_hz=safe_float(dpg.get_value("training.filter_high"), "High frequency"),
            ripple_db=safe_float(dpg.get_value("training.filter_ripple"), "Ripple"),
        )
        feature_config = FeatureConfig(
            use_ecg_time=bool(dpg.get_value("training.feat_ecg_time")),
            use_ecg_spectrum=bool(dpg.get_value("training.feat_ecg_spectrum")),
            use_acceleration=bool(dpg.get_value("training.feat_acc")),
            use_acceleration_diff=bool(dpg.get_value("training.feat_acc_diff")),
            use_demographics=bool(dpg.get_value("training.feat_demo")),
            spectrum_bins=safe_int(dpg.get_value("training.spectrum_bins"), "Spectrum bins"),
        )
        return DatasetBuildConfig(
            window_seconds=safe_float(dpg.get_value("training.window"), "Window seconds"),
            overlap=safe_float(dpg.get_value("training.overlap"), "Overlap"),
            label_source=dpg.get_value("training.labels"),
            max_windows_per_record=safe_int(dpg.get_value("training.max_windows"), "Maximum windows"),
            balance=dpg.get_value("training.balance"),
            filter_config=filter_config,
            feature_config=feature_config,
        )

    def _training_config(self) -> TrainingConfig:
        dpg = self.dpg
        metric_map = {
            "Loss": "loss",
            "Accuracy": "accuracy",
            "Balanced Accuracy": "balanced_accuracy",
            "Precision": "precision",
            "Recall": "recall",
            "F1": "f1",
        }
        scope_value = str(dpg.get_value("training.selection_scope"))
        return TrainingConfig(
            model_type=dpg.get_value("training.model_type"),
            validation_fraction=safe_float(dpg.get_value("training.validation_fraction"), "Validation fraction"),
            final_test_fraction=safe_float(dpg.get_value("training.final_test_fraction"), "Final-test fraction"),
            split_strategy=(
                "stratified_group_kfold"
                if dpg.get_value("training.split_strategy") == "Stratified Group K-Fold combinations"
                else "stratified_holdout"
            ),
            group_folds=safe_int(dpg.get_value("training.group_folds"), "Group folds"),
            random_seed=safe_int(dpg.get_value("training.seed"), "Random seed"),
            standardize=bool(dpg.get_value("training.standardize")),
            split_by_subject=bool(dpg.get_value("training.subject_split")),
            class_weight_balanced=bool(dpg.get_value("training.class_weight")),
            class_equalization=dpg.get_value("training.class_equalization"),
            equalize_validation=bool(
                dpg.get_value("training.equalize_validation")
            ),
            mlp_hidden_layers=dpg.get_value("training.mlp_layers"),
            mlp_epochs=safe_int(dpg.get_value("training.mlp_epochs"), "Epochs"),
            learning_rate=safe_float(dpg.get_value("training.learning_rate"), "Learning rate"),
            early_stopping=bool(dpg.get_value("training.early_stopping")),
            early_stopping_patience=safe_int(
                dpg.get_value("training.early_patience"), "Early-stopping patience"
            ),
            early_stopping_min_delta=safe_float(
                dpg.get_value("training.early_delta"), "Minimum validation improvement"
            ),
            selection_metric=metric_map[dpg.get_value("training.selection_metric")],
            selection_scope="overall" if scope_value == "Overall" else "class",
            target_class=0 if scope_value == "Overall" else int(scope_value.rsplit(" ", 1)[1]),
            error_penalty_enabled=bool(dpg.get_value("training.error_penalty")),
            dangerous_actual_class=3,
            dangerous_predicted_class=1,
            dangerous_error_penalty=safe_float(
                dpg.get_value("training.error_penalty_weight"), "Dangerous-error penalty"
            ),
            minimum_evaluation_samples=safe_int(
                dpg.get_value("training.minimum_samples"), "Minimum evaluation samples"
            ),
            rf_estimators=safe_int(dpg.get_value("training.rf_estimators"), "Trees"),
            rf_max_depth=safe_int(dpg.get_value("training.rf_depth"), "Tree depth"),
        )

    def _optimization_config(self) -> OptimizationConfig:
        dpg = self.dpg
        metric_map = {
            "Loss": "loss",
            "Accuracy": "accuracy",
            "Balanced Accuracy": "balanced_accuracy",
            "Precision": "precision",
            "Recall": "recall",
            "F1": "f1",
        }
        scope_value = str(dpg.get_value("optimization.selection_scope"))
        varied_features = [
            name
            for name, tag in (
                ("ecg_time", "optimization.feat_ecg_time"),
                ("ecg_spectrum", "optimization.feat_ecg_spectrum"),
                ("acceleration", "optimization.feat_acc"),
                ("acceleration_diff", "optimization.feat_acc_diff"),
                ("demographics", "optimization.feat_demo"),
            )
            if dpg.get_value(tag)
        ]
        model_types = [
            name
            for name, tag in (
                ("MLP", "optimization.model_mlp"),
                ("Logistic Regression", "optimization.model_logistic"),
                ("Random Forest", "optimization.model_rf"),
                ("Gradient Boosting", "optimization.model_gb"),
            )
            if dpg.get_value(tag)
        ]
        architectures = (
            parse_mlp_architectures(dpg.get_value("optimization.architectures"))
            if "MLP" in model_types
            else []
        )
        return OptimizationConfig(
            record_ids=self._selected_records(),
            window_lengths=parse_window_lengths(
                dpg.get_value("optimization.windows")
            ),
            varied_features=varied_features,
            model_types=model_types,
            mlp_architectures=architectures,
            dataset_root=dpg.get_value("training.dataset_root"),
            database_root=dpg.get_value("optimization.database_root"),
            model_root=dpg.get_value("optimization.model_root"),
            run_name=dpg.get_value("optimization.run_name"),
            overlap=safe_float(dpg.get_value("training.overlap"), "Overlap"),
            label_source=dpg.get_value("training.labels"),
            max_windows_per_record=safe_int(
                dpg.get_value("training.max_windows"), "Maximum windows"
            ),
            spectrum_bins=safe_int(
                dpg.get_value("training.spectrum_bins"), "Spectrum bins"
            ),
            validation_fraction=safe_float(
                dpg.get_value("optimization.validation_fraction"), "Validation fraction"
            ),
            final_test_fraction=safe_float(
                dpg.get_value("optimization.final_test_fraction"), "Final-test fraction"
            ),
            split_strategy=(
                "stratified_group_kfold"
                if dpg.get_value("optimization.split_strategy")
                == "Stratified Group K-Fold combinations"
                else "stratified_holdout"
            ),
            group_folds=safe_int(
                dpg.get_value("optimization.group_folds"), "Group folds"
            ),
            random_seed=safe_int(dpg.get_value("training.seed"), "Random seed"),
            split_by_subject=bool(dpg.get_value("optimization.subject_split")),
            standardize=bool(dpg.get_value("optimization.standardize")),
            class_equalization=dpg.get_value("optimization.equalization"),
            equalize_validation=bool(
                dpg.get_value("optimization.equalize_validation")
            ),
            class_weight_balanced=bool(dpg.get_value("training.class_weight")),
            mlp_epochs=safe_int(
                dpg.get_value("optimization.epochs"), "Maximum epochs"
            ),
            learning_rate=safe_float(
                dpg.get_value("training.learning_rate"), "Learning rate"
            ),
            early_stopping_patience=safe_int(
                dpg.get_value("optimization.patience"),
                "Early-stopping patience",
            ),
            early_stopping_min_delta=safe_float(
                dpg.get_value("optimization.min_delta"),
                "Minimum validation improvement",
            ),
            selection_metric=metric_map[dpg.get_value("optimization.selection_metric")],
            selection_scope="overall" if scope_value == "Overall" else "class",
            target_class=0 if scope_value == "Overall" else int(scope_value.rsplit(" ", 1)[1]),
            error_penalty_enabled=bool(
                dpg.get_value("optimization.error_penalty")
            ),
            dangerous_error_penalty=safe_float(
                dpg.get_value("optimization.error_penalty_weight"),
                "Dangerous-error penalty",
            ),
            minimum_evaluation_samples=safe_int(
                dpg.get_value("optimization.minimum_samples"),
                "Minimum evaluation samples",
            ),
            rf_estimators=safe_int(
                dpg.get_value("training.rf_estimators"), "Trees"
            ),
            rf_max_depth=safe_int(
                dpg.get_value("training.rf_depth"), "Tree depth"
            ),
            max_parallel_workers=safe_int(
                dpg.get_value("optimization.max_workers"), "Parallel workers"
            ),
            ram_fraction=safe_int(
                dpg.get_value("optimization.ram_fraction"), "RAM budget"
            )
            / 100.0,
        )

    def _preview_optimization(self, *_args) -> None:
        try:
            config = self._optimization_config()
            config.validate()
            datasets, models = optimization_size(config)
            self.dpg.set_value(
                "optimization.summary",
                f"{datasets:,} prepared datasets and {models:,} trained model candidates. "
                f"Only {len(config.window_lengths):,} signal/feature-superset build(s) are required; feature datasets are derived from them. "
                "CPU/RAM concurrency is automatic; every candidate uses the same scikit-learn training path as Train Models.",
            )
            self.dpg.configure_item("optimization.summary", color=GOOD)
        except Exception as exc:
            self.dpg.set_value("optimization.summary", str(exc))
            self.dpg.configure_item("optimization.summary", color=DANGER)

    def _start_optimization(self, *_args) -> None:
        try:
            if not self.repository.records:
                self._scan()
            config = self._optimization_config()
            config.validate()
            datasets, models = optimization_size(config)
            self._begin_work(
                f"Optimizing {models:,} candidates across {datasets:,} datasets"
            )

            def work() -> None:
                try:
                    result = run_optimization(
                        self.repository,
                        config,
                        progress=lambda value, text: self.events.put(
                            ("progress", (value, text))
                        ),
                        cancel=self.cancel,
                    )
                    self.events.put(("optimization_ready", result))
                except Exception as exc:
                    self.events.put(("error", str(exc)))

            self.worker = threading.Thread(
                target=work, name="model-optimization", daemon=True
            )
            self.worker.start()
        except Exception as exc:
            self._error(str(exc))

    def _build_dataset(self, *_args) -> None:
        try:
            if not self.repository.records:
                self._scan()
            self.build_config = self._dataset_config()
            record_ids = self._selected_records()
            self._begin_work("Preparing dataset")

            def work() -> None:
                try:
                    frame = build_dataset(
                        self.repository,
                        record_ids,
                        self.build_config,
                        progress=lambda value, text: self.events.put(("progress", (value, text))),
                        cancel=self.cancel,
                    )
                    self.events.put(("dataset_ready", frame))
                except Exception as exc:
                    self.events.put(("error", str(exc)))

            self.worker = threading.Thread(target=work, name="dataset-builder", daemon=True)
            self.worker.start()
        except Exception as exc:
            self._error(str(exc))

    def _train(self, *_args) -> None:
        try:
            if self.frame is None or self.frame.empty:
                raise ValueError("Build or load a prepared dataset first")
            config = self._training_config()
            self._begin_work(f"Training {config.model_type}")

            def work() -> None:
                try:
                    result = train_model(
                        self.frame,
                        config,
                        progress=lambda value, text: self.events.put(("progress", (value, text))),
                        cancel=self.cancel,
                    )
                    self.events.put(("model_ready", result))
                except Exception as exc:
                    self.events.put(("error", str(exc)))

            self.worker = threading.Thread(target=work, name="model-training", daemon=True)
            self.worker.start()
        except Exception as exc:
            self._error(str(exc))

    def _begin_work(self, message: str) -> None:
        if self.worker and self.worker.is_alive():
            raise RuntimeError("Another operation is already running")
        self.cancel.clear()
        for tag in ("training.build", "training.train", "optimization.start"):
            self.dpg.configure_item(tag, enabled=False)
        for tag in ("training.cancel", "optimization.cancel"):
            self.dpg.configure_item(tag, enabled=True)
        self.dpg.set_value("training.status", message)
        self.dpg.configure_item("training.status", color=(245, 174, 66, 255))

    def _cancel(self, *_args) -> None:
        self.cancel.set()

    def _work_finished(self) -> None:
        for tag in ("training.build", "training.train", "optimization.start"):
            self.dpg.configure_item(tag, enabled=True)
        for tag in ("training.cancel", "optimization.cancel"):
            self.dpg.configure_item(tag, enabled=False)

    def _load_selected(self, _sender, app_data) -> None:
        path = app_data.get("file_path_name")
        if not path:
            return
        try:
            self.frame = load_dataframe(path)
            self.result = None
            self.dpg.configure_item("training.export", enabled=False)
            source = Path(path).expanduser().resolve()
            sidecar = source.with_suffix(source.suffix + ".config.json")
            if sidecar.is_file():
                values = json.loads(sidecar.read_text(encoding="utf-8"))
                values["filter_config"] = FilterConfig(**values.get("filter_config", {}))
                values["feature_config"] = FeatureConfig(**values.get("feature_config", {}))
                self.build_config = DatasetBuildConfig(**values)
            else:
                self.build_config = self._dataset_config()
                self._log(
                    "No dataset configuration sidecar was found; export will use the current preparation settings."
                )
            self._show_dataset_summary()
            self._log(f"Loaded prepared dataset: {path}")
        except Exception as exc:
            self._error(str(exc))

    def _save_selected(self, _sender, app_data) -> None:
        path = app_data.get("file_path_name")
        try:
            if self.frame is None:
                raise ValueError("No prepared dataset to save")
            target = save_dataframe(self.frame, path)
            if self.build_config is None:
                self.build_config = self._dataset_config()
            target.with_suffix(target.suffix + ".config.json").write_text(
                json.dumps(asdict(self.build_config), indent=2), encoding="utf-8"
            )
            self._log(f"Saved prepared dataset: {target}")
        except Exception as exc:
            self._error(str(exc))

    def _export_selected(self, _sender, app_data) -> None:
        directory = app_data.get("file_path_name") or app_data.get("current_path")
        try:
            if self.result is None or self.build_config is None or self.frame is None:
                raise ValueError("Train a model before exporting it")
            folder = Path(directory).expanduser().resolve()
            folder.mkdir(parents=True, exist_ok=True)
            model_path = save_model_bundle(
                folder / "model_bundle.joblib",
                self.result.estimator,
                self.result.feature_columns,
                self.build_config.filter_config,
                self.build_config.feature_config,
                self.result.metrics,
                {
                    "training": asdict(self.result.config),
                    "dataset": asdict(self.build_config),
                },
            )
            save_training_report(self.result, folder / "training_results.json")
            exported = export_training_artifacts(
                self.result,
                self.build_config,
                self.frame,
                folder,
            )
            self._log(f"Exported server-ready model: {model_path}")
            self._log(
                "Exported report: "
                + ", ".join(path.name for path in exported.values())
            )
        except Exception as exc:
            self._error(str(exc))

    def _poll_events(self) -> None:
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "progress":
                value, text = payload
                self.dpg.set_value("training.progress", float(value))
                self.dpg.configure_item("training.progress", overlay=f"{float(value):.0%}")
                self.dpg.set_value("training.status", str(text))
            elif kind == "dataset_ready":
                self.frame = payload
                self.result = None
                self.dpg.configure_item("training.export", enabled=False)
                self._show_dataset_summary()
                self._work_finished()
                self.dpg.set_value("training.status", "Dataset ready")
                self.dpg.configure_item("training.status", color=GOOD)
            elif kind == "model_ready":
                self.result = payload
                self._show_results()
                self.dpg.configure_item("training.export", enabled=True)
                self._work_finished()
                self.dpg.set_value("training.status", "Model ready")
                self.dpg.configure_item("training.status", color=GOOD)
            elif kind == "optimization_ready":
                self.optimization_run = payload
                self._show_optimization_results()
                self._work_finished()
                self.dpg.set_value("training.status", "Optimization complete")
                self.dpg.configure_item("training.status", color=GOOD)
            elif kind == "error":
                self._work_finished()
                self._error(str(payload))

    def _show_optimization_results(self) -> None:
        assert self.optimization_run is not None
        self.dpg.delete_item("optimization.results", children_only=True, slot=1)
        for row_index, row in enumerate(self.optimization_run.rows):
            accuracy = row.get("accuracy")
            balanced = row.get("balanced_accuracy")
            macro_f1 = row.get("macro_f1")
            seconds = row.get("training_seconds")
            values = (
                row.get("best_for", ""),
                row.get("status", ""),
                f"{float(row.get('window_seconds', 0)):g}s",
                row.get("features", ""),
                row.get("model", ""),
                row.get("training_backend", ""),
                row.get("architecture", ""),
                _metric_text(accuracy),
                _metric_text(balanced),
                _metric_text(macro_f1),
                row.get("best_epoch", ""),
                row.get("epochs_completed", ""),
                _metric_text(seconds, 2),
                row.get("candidate")
                if row.get("status") == "ok"
                else row.get("error", ""),
            )
            with self.dpg.table_row(parent="optimization.results"):
                for value in values:
                    self.dpg.add_text(str(value))
            if row.get("best_for"):
                self.dpg.highlight_table_row(
                    "optimization.results", row_index, (18, 124, 153, 95)
                )
        self.dpg.set_value(
            "optimization.summary",
            f"Run {self.optimization_run.run_name}: {len(self.optimization_run.rows):,} result rows. "
            f"Models: {self.optimization_run.model_folder} | Datasets: {self.optimization_run.database_folder}\n"
            f"{self.optimization_run.resource_summary}",
        )
        self.dpg.configure_item("optimization.summary", color=GOOD)

    def _show_dataset_summary(self) -> None:
        assert self.frame is not None
        summary = dataset_summary(self.frame)
        self.dpg.set_value(
            "training.dataset_summary",
            f"{summary['windows']:,} windows | {summary['records']} recordings | "
            f"{summary['subjects']} people | {summary['features']} features"
            + (
                f" | capped {len(summary['capped_records'])} recording(s) at "
                f"{summary['max_windows_per_record']:,}; "
                f"{summary['eligible_windows']:,} windows were eligible"
                if summary["capped_records"]
                else " | no per-record cap applied"
            ),
        )
        replace_table_rows(
            self.dpg,
            "training.class_table",
            ((quality_class, count) for quality_class, count in summary["class_counts"].items()),
        )
        self._log(f"Prepared dataset: {summary}")

    def _show_results(self) -> None:
        assert self.result is not None
        metrics = self.result.metrics
        self.dpg.set_value(
            "training.metric_accuracy", f"{self.result.metrics['accuracy']:.3f}"
        )
        self.dpg.set_value(
            "training.metric_f1", f"{self.result.metrics['f1_macro']:.3f}"
        )
        self.dpg.set_value(
            "training.metric_train", f"{self.result.metrics['train_windows']:,}"
        )
        self.dpg.set_value(
            "training.metric_test",
            f"{self.result.metrics.get('final_test_windows', self.result.metrics.get('test_windows', 0)):,}",
        )
        replace_table_rows(
            self.dpg,
            "training.overall_table",
            (
                ("Accuracy", _metric_with_ci(metrics["accuracy"], metrics, "accuracy")),
                ("Balanced accuracy", _metric_with_ci(metrics["balanced_accuracy"], metrics, "balanced_accuracy")),
                ("Macro precision", _metric_with_ci(metrics["precision_macro"], metrics, "precision_macro")),
                ("Macro recall", _metric_with_ci(metrics["recall_macro"], metrics, "recall_macro")),
                ("Macro F1", _metric_with_ci(metrics["f1_macro"], metrics, "macro_f1")),
                ("Weighted F1", f"{metrics['f1_weighted']:.4f}"),
                ("Input features", len(self.result.feature_columns)),
                ("Training backend", metrics.get("training_backend", "-")),
                ("Epochs completed", metrics.get("epochs_completed", "-")),
                ("Best epoch", metrics.get("best_epoch", "-")),
                ("Early stopped", "yes" if metrics.get("early_stopped") else "no"),
                ("Selection target", metrics.get("selection_target", metrics.get("selection_metric", "-"))),
                ("Validation objective", f"{metrics.get('validation_objective_score', 0.0):.4f}"),
                ("Dangerous 3 -> 1 errors", metrics.get("dangerous_error_count", 0)),
                ("Dangerous 3 -> 1 rate among Class 3", f"{100.0 * float(metrics.get('dangerous_error_rate', 0.0)):.2f}%"),
                ("Evaluation combinations", metrics.get("evaluation_candidate_count", 1)),
                ("Data preparation time", _duration_metric(metrics.get("data_preparation_seconds"))),
                ("Total training time", _duration_metric(metrics.get("training_seconds"))),
            ),
        )
        before_train = metrics.get("train_class_counts_before", {})
        after_train = metrics.get("train_class_counts_after", {})
        before_validation = metrics.get("validation_class_counts_before", {})
        after_validation = metrics.get("validation_class_counts_after", {})
        final_counts = metrics.get("test_class_counts", {})
        replace_table_rows(
            self.dpg,
            "training.balance_table",
            (
                (
                    quality_class,
                    _mapping_count(before_train, quality_class),
                    _mapping_count(after_train, quality_class),
                    _mapping_count(before_validation, quality_class),
                    _mapping_count(after_validation, quality_class),
                    _mapping_count(final_counts, quality_class),
                )
                for quality_class in self.result.classes
            ),
        )
        classification = metrics.get("classification_report", {})
        intervals = metrics.get("confidence_intervals_95", {})
        classification_rows = []
        for label in [str(value) for value in self.result.classes] + [
            "macro avg",
            "weighted avg",
        ]:
            values = classification.get(label)
            if isinstance(values, dict):
                classification_rows.append(
                    (
                        label,
                        f"{float(values.get('precision', 0.0)):.3f}",
                        f"{float(values.get('recall', 0.0)):.3f}",
                        f"{float(values.get('f1-score', 0.0)):.3f}",
                        f"{float(values.get('support', 0.0)):g}",
                        _class_interval_text(intervals, label),
                    )
                )
        replace_table_rows(
            self.dpg, "training.classification_table", classification_rows
        )
        warnings = metrics.get("evaluation_warnings", [])
        self.dpg.set_value(
            "training.evaluation_warning",
            "\n".join(f"WARNING: {value}" for value in warnings)
            if warnings
            else "All validation and final-test classes meet the configured minimum sample count.",
        )
        self.dpg.configure_item(
            "training.evaluation_warning", color=WARNING if warnings else GOOD
        )
        self.dpg.set_value(
            "training.subject_summary",
            f"Train subjects ({len(metrics.get('train_subjects', []))}): "
            + ", ".join(metrics.get("train_subjects", []))
            + f"\nValidation subjects ({len(metrics.get('validation_subjects', []))}): "
            + ", ".join(metrics.get("validation_subjects", []))
            + f"\nFinal-test subjects ({len(metrics.get('final_test_subjects', []))}): "
            + ", ".join(metrics.get("final_test_subjects", [])),
        )
        replace_table_rows(
            self.dpg,
            "training.subject_allocation_table",
            (
                (
                    row["subject_id"],
                    row["total"],
                    row["train_count"],
                    f"{row['train_percent']:.1f}%",
                    row["validation_count"],
                    f"{row['validation_percent']:.1f}%",
                    row["final_test_count"],
                    f"{row['final_test_percent']:.1f}%",
                )
                for row in metrics.get("subject_allocation", [])
            ),
        )
        replace_table_rows(
            self.dpg,
            "training.evaluation_candidates_table",
            (
                (
                    row.get("candidate", "-"),
                    "Yes" if row.get("selected") else "",
                    _metric_text(row.get("validation_objective_score")),
                    _metric_text(row.get("final_accuracy")),
                    _metric_text(row.get("final_balanced_accuracy")),
                    _metric_text(row.get("final_macro_f1")),
                    row.get("dangerous_error_count", 0),
                    ", ".join(row.get("train_subjects", [])),
                    ", ".join(row.get("validation_subjects", [])),
                    ", ".join(row.get("final_test_subjects", [])),
                )
                for row in metrics.get("evaluation_candidates", [])
            ),
        )
        epochs = list(range(1, len(self.result.loss_curve) + 1))
        self.dpg.set_value("training.loss_series", [epochs, self.result.loss_curve])
        self.dpg.set_value("training.accuracy_series", [epochs, self.result.accuracy_curve])
        if epochs:
            self.dpg.fit_axis_data("training.loss_x")
            self.dpg.fit_axis_data("training.loss_y")
            self.dpg.fit_axis_data("training.accuracy_x")
            if metrics.get("selection_metric") != "loss":
                self.dpg.set_axis_limits("training.accuracy_y", 0.0, 1.0)
            else:
                self.dpg.fit_axis_data("training.accuracy_y")
        objective_label = str(
            metrics.get("selection_target", metrics.get("selection_metric", "accuracy"))
        )
        self.dpg.configure_item("training.objective_plot", label=f"Validation {objective_label}")
        self.dpg.configure_item("training.accuracy_y", label=objective_label)
        rgba = render_confusion_matrix_rgba(self.result, 720, 620)
        self.dpg.set_value(
            "training.confusion_texture",
            (rgba.astype(np.float32) / 255.0).reshape(-1).tolist(),
        )
        self._log(f"Training finished: accuracy={self.result.metrics['accuracy']:.4f}")

    def _log(self, message: str) -> None:
        current = self.dpg.get_value("training.log") if self.dpg.does_item_exist("training.log") else ""
        lines = (current.splitlines() + [message])[-100:]
        if self.dpg.does_item_exist("training.log"):
            self.dpg.set_value("training.log", "\n".join(lines))

    def _error(self, message: str) -> None:
        self.dpg.set_value("training.status", message)
        self.dpg.configure_item("training.status", color=DANGER)
        self._log(f"ERROR: {message}")


def _mapping_count(mapping: dict, quality_class: int) -> int:
    return int(mapping.get(quality_class, mapping.get(str(quality_class), 0)))


def _metric_text(value: object, decimals: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{number:.{decimals}f}" if math.isfinite(number) else "-"


def _metric_with_ci(value: object, metrics: dict, key: str) -> str:
    text = _metric_text(value)
    interval = metrics.get("confidence_intervals_95", {}).get(key)
    if isinstance(interval, (list, tuple)) and len(interval) == 2:
        text += f" (95% CI {float(interval[0]):.4f}-{float(interval[1]):.4f})"
    return text


def _class_interval_text(intervals: dict, label: str) -> str:
    if label == "macro avg":
        interval = intervals.get("macro_f1")
        return (
            f"F1 {float(interval[0]):.3f}-{float(interval[1]):.3f}"
            if isinstance(interval, (list, tuple)) and len(interval) == 2
            else "-"
        )
    values = intervals.get("per_class", {}).get(str(label), {})
    parts = []
    for name in ("precision", "recall", "f1"):
        interval = values.get(name)
        if isinstance(interval, (list, tuple)) and len(interval) == 2:
            parts.append(f"{name[0].upper()} {interval[0]:.3f}-{interval[1]:.3f}")
    return "; ".join(parts) or "-"


def _duration_metric(value: object) -> str:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{seconds:.2f} s"


def main() -> None:
    TrainingApp().run()


if __name__ == "__main__":
    main()
