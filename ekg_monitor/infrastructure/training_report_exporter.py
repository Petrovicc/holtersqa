from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ekg_monitor.application.models import DatasetBuildConfig
from ekg_monitor.application.training_service import TrainingResult, dataset_summary


NAVY = "102A36"
TEAL = "13799B"
PALE_BLUE = "DCEEF4"
PALE_GREEN = "E4F3ED"
WHITE = "FFFFFF"
TEXT = "263B46"
GRID = "CFDDE3"


def export_training_artifacts(
    result: TrainingResult,
    build_config: DatasetBuildConfig,
    frame: pd.DataFrame,
    directory: str | Path,
) -> dict[str, Path]:
    """Export model evaluation images and a three-sheet XLSX report."""
    folder = Path(directory).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    paths = {
        "loss": folder / "training_loss.png",
        "accuracy": folder / "validation_accuracy.png",
        "confusion_matrix": folder / "confusion_matrix.png",
        "workbook": folder / "training_report.xlsx",
    }
    _export_curve(
        result.loss_curve,
        paths["loss"],
        title="Training Loss",
        y_label="Loss",
        color="#D65B70",
    )
    objective_name = str(
        result.metrics.get(
            "selection_target", result.metrics.get("selection_metric", "accuracy")
        )
    ).replace("_", " ").title()
    _export_curve(
        result.accuracy_curve,
        paths["accuracy"],
        title=f"Validation {objective_name}",
        y_label=objective_name,
        color="#15906D",
        y_limits=None if str(result.metrics.get("selection_metric")) == "loss" else (0.0, 1.0),
    )
    _export_confusion_matrix(result, paths["confusion_matrix"])
    _export_workbook(result, build_config, frame, paths["workbook"])
    return paths


def _export_curve(
    values: list[float],
    path: Path,
    *,
    title: str,
    y_label: str,
    color: str,
    y_limits: tuple[float, float] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(9.5, 5.4), dpi=160)
    figure.patch.set_facecolor("#F7FAFB")
    axis.set_facecolor("#FFFFFF")
    if values:
        epochs = np.arange(1, len(values) + 1)
        axis.plot(epochs, values, color=color, linewidth=2.2)
        axis.fill_between(epochs, values, alpha=0.10, color=color)
    else:
        axis.text(
            0.5,
            0.5,
            "This model does not expose an epoch-by-epoch curve.",
            ha="center",
            va="center",
            transform=axis.transAxes,
            color="#526A76",
        )
    axis.set_title(title, fontsize=16, weight="bold", color="#173846", pad=14)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(y_label)
    if y_limits:
        axis.set_ylim(*y_limits)
    axis.grid(True, color="#D9E5E9", linewidth=0.8, alpha=0.9)
    for spine in axis.spines.values():
        spine.set_color("#B9CDD5")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _export_confusion_matrix(result: TrainingResult, path: Path) -> None:
    figure = _confusion_figure(result, 900, 760)
    figure.savefig(path, facecolor="#F7FAFB", dpi=120)
    from matplotlib import pyplot as plt

    plt.close(figure)


def render_confusion_matrix_rgba(
    result: TrainingResult, width: int = 720, height: int = 620
) -> np.ndarray:
    """Render the exact export-style matrix for a Dear PyGui texture."""
    figure = _confusion_figure(result, width, height)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib import pyplot as plt

    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8).copy()
    plt.close(figure)
    return rgba


def render_confusion_matrix_array_rgba(
    matrix: np.ndarray,
    classes: list[int],
    title: str,
    width: int = 620,
    height: int = 540,
) -> np.ndarray:
    """Render count and row-percentage cells without GUI annotation widgets."""
    figure = _confusion_array_figure(matrix, classes, title, width, height)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib import pyplot as plt

    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8).copy()
    plt.close(figure)
    return rgba


def _confusion_figure(result: TrainingResult, width: int, height: int):
    return _confusion_array_figure(
        np.asarray(result.confusion_matrix, dtype=int),
        [int(value) for value in result.classes],
        "Confusion Matrix",
        width,
        height,
    )


def _confusion_array_figure(
    matrix: np.ndarray,
    class_values: list[int],
    title: str,
    width: int,
    height: int,
):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    matrix = np.asarray(matrix, dtype=int)
    classes = [str(value) for value in class_values]
    dpi = 120
    figure, axis = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    figure.patch.set_facecolor("#F7FAFB")
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Windows")
    axis.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted quality class",
        ylabel="Actual quality class",
        title=title,
    )
    axis.title.set_fontsize(16)
    axis.title.set_weight("bold")
    threshold = float(matrix.max()) / 2.0 if matrix.size else 0.0
    row_totals = matrix.sum(axis=1)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            percentage = (
                100.0 * matrix[row, column] / float(row_totals[row])
                if row_totals[row]
                else 0.0
            )
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}\n{percentage:.1f}%",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "#173846",
                weight="bold",
            )
    figure.tight_layout(pad=1.1)
    return figure


def _export_workbook(
    result: TrainingResult,
    build_config: DatasetBuildConfig,
    frame: pd.DataFrame,
    path: Path,
) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

    workbook = Workbook()
    performance = workbook.active
    performance.title = "Model Performance"
    parameters = workbook.create_sheet("Model Parameters")
    preparation = workbook.create_sheet("Data Preparation")

    _write_performance_sheet(performance, result)
    _write_parameter_sheet(parameters, result)
    _write_preparation_sheet(preparation, build_config, frame)
    parameters.auto_filter.ref = f"A4:C{parameters.max_row}"
    preparation.auto_filter.ref = f"A4:C{preparation.max_row}"

    thin = Side(style="thin", color=GRID)
    for sheet in workbook.worksheets:
        sheet.sheet_view.showGridLines = False
        sheet.freeze_panes = "A5"
        for row in sheet.iter_rows():
            for cell in row:
                fill_color = str(cell.fill.fgColor.rgb or "")[-6:]
                if fill_color == TEAL:
                    font_color, bold = WHITE, True
                elif fill_color == PALE_BLUE:
                    font_color, bold = NAVY, True
                else:
                    font_color, bold = TEXT, bool(cell.font.bold)
                cell.font = Font(
                    name="Aptos", size=10, color=font_color, bold=bold
                )
                cell.alignment = Alignment(
                    vertical="top",
                    horizontal="right" if isinstance(cell.value, (int, float)) else "left",
                    wrap_text=True,
                )
        sheet["A1"].font = Font(name="Aptos Display", size=18, bold=True, color=WHITE)
        sheet["A1"].fill = PatternFill("solid", fgColor=NAVY)
        for cell in sheet[1]:
            cell.fill = PatternFill("solid", fgColor=NAVY)
        for row in sheet.iter_rows():
            if row and row[0].value and all(cell.fill.fill_type is None for cell in row):
                for cell in row:
                    cell.border = Border(bottom=thin)
        for row in sheet.iter_rows(min_row=4):
            label = str(row[0].value or "").lower()
            if len(row) > 1 and isinstance(row[1].value, float) and any(
                token in label
                for token in ("accuracy", "precision", "recall", "f1")
            ):
                row[1].number_format = "0.0000"
        _fit_columns(sheet)
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)


def _write_performance_sheet(sheet, result: TrainingResult) -> None:
    sheet.append(["ECG Model Performance Report", None, None, None, None])
    sheet.append(["Model", result.config.model_type])
    sheet.append([])
    _append_header(sheet, ["Metric", "Value"])
    metric_keys = (
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "selection_metric",
        "selection_scope",
        "selection_target",
        "target_class",
        "validation_objective_score",
        "dangerous_error_penalty_enabled",
        "dangerous_actual_class",
        "dangerous_predicted_class",
        "dangerous_error_penalty",
        "dangerous_error_count",
        "dangerous_error_rate",
        "train_windows_original",
        "train_windows",
        "validation_windows_original",
        "validation_windows",
        "final_test_windows_original",
        "final_test_windows",
        "class_equalization",
        "validation_equalized",
        "epochs_requested",
        "epochs_completed",
        "best_epoch",
        "early_stopping_enabled",
        "early_stopped",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "training_backend",
        "evaluation_candidate_count",
        "selected_evaluation_candidate",
        "data_preparation_seconds",
        "training_seconds",
    )
    for key in metric_keys:
        sheet.append([_label(key), _excel_value(result.metrics.get(key))])
    sheet.append([])
    _append_section(sheet, "Class distribution")
    _append_header(
        sheet,
        [
            "Quality class",
            "Train before",
            "Train used",
            "Validation before",
            "Validation used",
            "Final test",
        ],
    )
    before = result.metrics.get("train_class_counts_before", {})
    after = result.metrics.get("train_class_counts_after", {})
    validation_before = result.metrics.get("validation_class_counts_before", {})
    validation_after = result.metrics.get("validation_class_counts_after", {})
    final_counts = result.metrics.get(
        "test_class_counts_after", result.metrics.get("test_class_counts", {})
    )
    for quality_class in result.classes:
        sheet.append(
            [
                quality_class,
                _mapping_value(before, quality_class),
                _mapping_value(after, quality_class),
                _mapping_value(validation_before, quality_class),
                _mapping_value(validation_after, quality_class),
                _mapping_value(final_counts, quality_class),
            ]
        )
    sheet.append([])
    _append_section(sheet, "Per-class performance")
    _append_header(
        sheet,
        [
            "Class / average",
            "Precision",
            "Precision 95% CI",
            "Recall",
            "Recall 95% CI",
            "F1 score",
            "F1 95% CI",
            "Support",
        ],
    )
    report = result.metrics.get("classification_report", {})
    intervals = result.metrics.get("confidence_intervals_95", {})
    per_class_intervals = intervals.get("per_class", {})
    report_order = [str(value) for value in result.classes] + ["macro avg", "weighted avg"]
    for key in report_order:
        values = report.get(key)
        if isinstance(values, dict):
            class_intervals = per_class_intervals.get(key, {})
            sheet.append(
                [
                    key,
                    _excel_value(values.get("precision")),
                    _interval_value(class_intervals.get("precision")),
                    _excel_value(values.get("recall")),
                    _interval_value(class_intervals.get("recall")),
                    _excel_value(values.get("f1-score")),
                    _interval_value(
                        intervals.get("macro_f1")
                        if key == "macro avg"
                        else class_intervals.get("f1")
                    ),
                    _excel_value(values.get("support")),
                ]
            )
    sheet.append([])
    _append_section(sheet, "95% confidence intervals")
    _append_header(sheet, ["Metric", "Interval"])
    sheet.append(["Accuracy", _interval_value(intervals.get("accuracy"))])
    sheet.append(
        ["Balanced accuracy", _interval_value(intervals.get("balanced_accuracy"))]
    )
    sheet.append(
        ["Macro precision", _interval_value(intervals.get("precision_macro"))]
    )
    sheet.append(["Macro recall", _interval_value(intervals.get("recall_macro"))])
    sheet.append(["Macro F1", _interval_value(intervals.get("macro_f1"))])
    sheet.append([])
    _append_section(sheet, "Subject allocation")
    _append_header(sheet, ["Partition", "Subjects"])
    sheet.append(["Train", _excel_value(result.metrics.get("train_subjects", []))])
    sheet.append(
        ["Validation", _excel_value(result.metrics.get("validation_subjects", []))]
    )
    sheet.append(
        ["Final test", _excel_value(result.metrics.get("final_test_subjects", []))]
    )
    sheet.append([])
    _append_header(
        sheet,
        [
            "Subject",
            "Total",
            "Train count",
            "Train %",
            "Validation count",
            "Validation %",
            "Final-test count",
            "Final-test %",
        ],
    )
    for row in result.metrics.get("subject_allocation", []):
        sheet.append(
            [
                row.get("subject_id"),
                row.get("total"),
                row.get("train_count"),
                row.get("train_percent"),
                row.get("validation_count"),
                row.get("validation_percent"),
                row.get("final_test_count"),
                row.get("final_test_percent"),
            ]
        )
    candidates = result.metrics.get("evaluation_candidates", [])
    if candidates:
        sheet.append([])
        _append_section(sheet, "Evaluated subject-fold combinations")
        _append_header(
            sheet,
            [
                "Candidate",
                "Selected",
                "Validation objective",
                "Final accuracy",
                "Final balanced accuracy",
                "Final macro F1",
                "Dangerous Class 3 -> 1",
                "Train subjects",
                "Validation subjects",
                "Final-test subjects",
            ],
        )
        for candidate in candidates:
            sheet.append(
                [
                    candidate.get("candidate"),
                    bool(candidate.get("selected")),
                    candidate.get("validation_objective_score"),
                    candidate.get("final_accuracy"),
                    candidate.get("final_balanced_accuracy"),
                    candidate.get("final_macro_f1"),
                    candidate.get("dangerous_error_count"),
                    _excel_value(candidate.get("train_subjects", [])),
                    _excel_value(candidate.get("validation_subjects", [])),
                    _excel_value(candidate.get("final_test_subjects", [])),
                ]
            )
    warnings = result.metrics.get("evaluation_warnings", [])
    if warnings:
        sheet.append([])
        _append_section(sheet, "Statistical warnings")
        for warning in warnings:
            sheet.append(["Warning", str(warning)])
    sheet.append([])
    _append_section(sheet, "Confusion matrix (rows = actual, columns = predicted)")
    _append_header(sheet, ["Actual / predicted", *[f"Class {value}" for value in result.classes]])
    matrix = np.asarray(result.confusion_matrix, dtype=int)
    for row_index, quality_class in enumerate(result.classes):
        sheet.append([f"Class {quality_class}", *[int(value) for value in matrix[row_index]]])


def _write_parameter_sheet(sheet, result: TrainingResult) -> None:
    sheet.append(["Model Parameters", None, None])
    sheet.append(["Model type", result.config.model_type])
    sheet.append([])
    _append_header(sheet, ["Section", "Parameter", "Value"])
    for key, value in _flatten_mapping(asdict(result.config)):
        sheet.append(["Training configuration", key, _excel_value(value)])
    estimator = result.estimator
    parameters = estimator.get_params(deep=True) if hasattr(estimator, "get_params") else {}
    for key, value in sorted(parameters.items()):
        sheet.append(["Estimator", key, _excel_value(value)])
    sheet.append(["Features", "feature_count", len(result.feature_columns)])
    for index, feature in enumerate(result.feature_columns, start=1):
        sheet.append(["Features", f"feature_{index}", feature])


def _write_preparation_sheet(
    sheet, build_config: DatasetBuildConfig, frame: pd.DataFrame
) -> None:
    summary = dataset_summary(frame)
    sheet.append(["Data Preparation Parameters", None, None])
    sheet.append(["Prepared windows", int(summary["windows"])])
    sheet.append([])
    _append_header(sheet, ["Section", "Parameter", "Value"])
    for key in ("windows", "records", "subjects", "features"):
        sheet.append(["Dataset summary", key, _excel_value(summary[key])])
    for quality_class, count in summary["class_counts"].items():
        sheet.append(["Dataset class counts", f"class_{quality_class}", int(count)])
    for key, value in _flatten_mapping(asdict(build_config)):
        sheet.append(["Preparation configuration", key, _excel_value(value)])
    diagnostics = frame.attrs.get("build_diagnostics", {})
    for key, value in _flatten_mapping(diagnostics):
        sheet.append(["Preparation diagnostics", key, _excel_value(value)])
    if "record_id" in frame:
        for index, record_id in enumerate(sorted(frame["record_id"].astype(str).unique()), start=1):
            sheet.append(["Record selection", f"record_{index}", record_id])
    if "subject_id" in frame:
        for index, subject_id in enumerate(sorted(frame["subject_id"].astype(str).unique()), start=1):
            sheet.append(["Subject selection", f"subject_{index}", subject_id])


def _append_header(sheet, values: list[Any]) -> None:
    sheet.append(values)
    row = sheet.max_row
    from openpyxl.styles import Font, PatternFill

    for cell in sheet[row]:
        cell.fill = PatternFill("solid", fgColor=TEAL)
        cell.font = Font(name="Aptos", size=10, bold=True, color=WHITE)


def _append_section(sheet, title: str) -> None:
    from openpyxl.styles import Font, PatternFill

    sheet.append([title])
    cell = sheet.cell(sheet.max_row, 1)
    cell.fill = PatternFill("solid", fgColor=PALE_BLUE)
    cell.font = Font(name="Aptos", size=11, bold=True, color=NAVY)


def _fit_columns(sheet) -> None:
    from openpyxl.utils import get_column_letter

    for column in range(1, sheet.max_column + 1):
        values = [sheet.cell(row, column).value for row in range(1, sheet.max_row + 1)]
        longest = max((len(str(value)) for value in values if value is not None), default=8)
        sheet.column_dimensions[get_column_letter(column)].width = min(58, max(13, longest + 2))
    for row in range(1, sheet.max_row + 1):
        sheet.row_dimensions[row].height = 20
    sheet.row_dimensions[1].height = 30


def _flatten_mapping(value: dict[str, Any], prefix: str = ""):
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            yield from _flatten_mapping(item, path)
        else:
            yield path, item


def _excel_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value)
    return str(value)


def _mapping_value(mapping: dict[Any, Any], key: int) -> int:
    return int(mapping.get(key, mapping.get(str(key), 0)))


def _interval_value(value: Any) -> str:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return ""
    return f"{float(value[0]):.4f} - {float(value[1]):.4f}"


def _label(value: str) -> str:
    return value.replace("_", " ").strip().title()
