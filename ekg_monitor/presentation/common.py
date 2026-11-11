from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


ACCENT = (20, 118, 158, 255)
GOOD = (25, 135, 84, 255)
WARNING = (190, 117, 16, 255)
DANGER = (190, 45, 55, 255)
MUTED = (100, 118, 132, 255)


def decimate(values: np.ndarray, maximum: int = 6000) -> tuple[list[float], list[float]]:
    array = np.asarray(values, dtype=float).reshape(-1)
    step = max(1, int(np.ceil(len(array) / maximum)))
    indices = np.arange(0, len(array), step, dtype=float)
    return indices.tolist(), array[::step].tolist()


def timed_axis(
    values: np.ndarray,
    fs: float,
    maximum: int = 6000,
    start_seconds: float = 0.0,
) -> tuple[list[float], list[float]]:
    x, y = decimate(values, maximum)
    if fs > 0:
        x = [start_seconds + value / fs for value in x]
    return x, y


def to_millivolts(values: np.ndarray, unit: str) -> np.ndarray:
    """Normalize WFDB physical ECG values to millivolts for clinical display."""
    normalized = str(unit or "").strip().lower().replace("micro", "u").replace("µ", "u")
    array = np.asarray(values, dtype=float)
    if normalized in {"uv", "uvolt", "uvolts"}:
        return array / 1000.0
    if normalized in {"v", "volt", "volts"}:
        return array * 1000.0
    return array


def safe_float(value: object, name: str) -> float:
    try:
        return float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc


def safe_int(value: object, name: str) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


class MedicalThemeController:
    """Dark-first medical theme and font manager shared by all applications."""

    def __init__(self, dpg, dark: bool = True) -> None:
        self.dpg = dpg
        self.dark = dark
        self._bindings: list[tuple[str, str]] = []
        self._load_fonts()
        self._build_global_theme("light", False)
        self._build_global_theme("dark", True)
        self._build_role_themes("light", False)
        self._build_role_themes("dark", True)
        self.apply()

    def _load_fonts(self) -> None:
        dpg = self.dpg
        regular = next(
            (
                path
                for path in (
                    Path("C:/Windows/Fonts/segoeui.ttf"),
                    Path("C:/Windows/Fonts/arial.ttf"),
                )
                if path.is_file()
            ),
            None,
        )
        semibold = next(
            (
                path
                for path in (
                    Path("C:/Windows/Fonts/seguisb.ttf"),
                    Path("C:/Windows/Fonts/arialbd.ttf"),
                )
                if path.is_file()
            ),
            regular,
        )
        if regular is None:
            return
        with dpg.font_registry():
            dpg.add_font(str(regular), 17, tag="medical.font.body")
            dpg.add_font(str(semibold), 20, tag="medical.font.heading")
            dpg.add_font(str(semibold), 27, tag="medical.font.metric")
        dpg.bind_font("medical.font.body")

    def _build_global_theme(self, name: str, dark: bool) -> None:
        dpg = self.dpg
        colors = (
            {
                "Text": (225, 235, 241, 255),
                "TextDisabled": (118, 139, 151, 255),
                "WindowBg": (15, 25, 32, 255),
                "ChildBg": (20, 34, 43, 255),
                "PopupBg": (20, 34, 43, 255),
                "Border": (53, 75, 88, 255),
                "FrameBg": (29, 47, 58, 255),
                "FrameBgHovered": (37, 62, 76, 255),
                "FrameBgActive": (43, 72, 87, 255),
                "Button": (19, 113, 150, 255),
                "ButtonHovered": (24, 139, 182, 255),
                "ButtonActive": (14, 93, 126, 255),
                "Header": (31, 70, 86, 255),
                "HeaderHovered": (38, 91, 110, 255),
                "HeaderActive": (27, 108, 140, 255),
                "Tab": (24, 43, 54, 255),
                "TabHovered": (31, 102, 129, 255),
                "TabActive": (20, 118, 158, 255),
                "TableHeaderBg": (27, 57, 70, 255),
                "TableRowBg": (20, 34, 43, 255),
                "TableRowBgAlt": (24, 41, 51, 255),
                "CheckMark": (69, 186, 218, 255),
                "SliderGrab": (31, 145, 183, 255),
                "Separator": (53, 75, 88, 255),
                "ScrollbarBg": (15, 25, 32, 255),
                "ScrollbarGrab": (64, 86, 99, 255),
            }
            if dark
            else {
                "Text": (31, 52, 65, 255),
                "TextDisabled": (121, 137, 146, 255),
                "WindowBg": (242, 247, 249, 255),
                "ChildBg": (255, 255, 255, 255),
                "PopupBg": (255, 255, 255, 255),
                "Border": (198, 216, 224, 255),
                "FrameBg": (231, 240, 244, 255),
                "FrameBgHovered": (217, 233, 239, 255),
                "FrameBgActive": (204, 225, 233, 255),
                "Button": (20, 118, 158, 255),
                "ButtonHovered": (15, 139, 183, 255),
                "ButtonActive": (12, 95, 130, 255),
                "Header": (220, 236, 242, 255),
                "HeaderHovered": (203, 227, 236, 255),
                "HeaderActive": (184, 218, 230, 255),
                "Tab": (224, 237, 242, 255),
                "TabHovered": (195, 224, 235, 255),
                "TabActive": (20, 118, 158, 255),
                "TableHeaderBg": (214, 232, 239, 255),
                "TableRowBg": (255, 255, 255, 255),
                "TableRowBgAlt": (243, 248, 250, 255),
                "CheckMark": (10, 115, 151, 255),
                "SliderGrab": (20, 118, 158, 255),
                "Separator": (198, 216, 224, 255),
                "ScrollbarBg": (235, 242, 245, 255),
                "ScrollbarGrab": (174, 197, 207, 255),
            }
        )
        with dpg.theme(tag=f"medical.theme.{name}"):
            with dpg.theme_component(dpg.mvAll):
                for color_name, value in colors.items():
                    target = getattr(dpg, f"mvThemeCol_{color_name}", None)
                    if target is not None:
                        dpg.add_theme_color(target, value)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 15, 14)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 10, 6)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 9, 8)
                dpg.add_theme_style(dpg.mvStyleVar_ItemInnerSpacing, 7, 5)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 7)
                dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 7)
                dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 7)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 9)
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 9)
                dpg.add_theme_style(dpg.mvStyleVar_ScrollbarRounding, 8)
            with dpg.theme_component(dpg.mvPlot):
                plot_colors = (
                    {
                        "PlotBg": (14, 25, 31, 255),
                        "FrameBg": (14, 25, 31, 255),
                        "PlotBorder": (53, 75, 88, 255),
                        "AxisText": (180, 199, 209, 255),
                        "AxisGrid": (55, 75, 86, 80),
                        "LegendBg": (20, 34, 43, 230),
                        "LegendText": (225, 235, 241, 255),
                    }
                    if dark
                    else {
                        "PlotBg": (255, 252, 252, 255),
                        "FrameBg": (255, 255, 255, 255),
                        "PlotBorder": (203, 216, 222, 255),
                        "AxisText": (63, 82, 94, 255),
                        "AxisGrid": (210, 221, 226, 80),
                        "LegendBg": (255, 255, 255, 235),
                        "LegendText": (31, 52, 65, 255),
                    }
                )
                for color_name, value in plot_colors.items():
                    target = getattr(dpg, f"mvPlotCol_{color_name}", None)
                    if target is not None:
                        dpg.add_theme_color(target, value, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_PlotPadding,
                    12,
                    12,
                    category=dpg.mvThemeCat_Plots,
                )

    def _build_role_themes(self, name: str, dark: bool) -> None:
        dpg = self.dpg
        palette = (
            {
                "raw": (89, 169, 230, 255),
                "filtered": (43, 201, 153, 255),
                "acc_x": (89, 169, 230, 255),
                "acc_y": (242, 174, 73, 255),
                "acc_z": (208, 117, 232, 255),
                "loss": (235, 111, 126, 255),
                "accuracy": (68, 190, 145, 255),
                "grid_minor": (120, 55, 63, 70),
                "grid_major": (178, 66, 77, 125),
            }
            if dark
            else {
                "raw": (50, 93, 148, 255),
                "filtered": (8, 132, 111, 255),
                "acc_x": (42, 102, 171, 255),
                "acc_y": (204, 117, 26, 255),
                "acc_z": (132, 75, 160, 255),
                "loss": (190, 58, 74, 255),
                "accuracy": (18, 137, 92, 255),
                "grid_minor": (239, 178, 183, 90),
                "grid_major": (219, 105, 115, 145),
            }
        )
        for role, color in palette.items():
            component = dpg.mvInfLineSeries if role.startswith("grid_") else dpg.mvLineSeries
            with dpg.theme(tag=f"medical.role.{name}.{role}"):
                with dpg.theme_component(component):
                    dpg.add_theme_color(
                        dpg.mvPlotCol_Line,
                        color,
                        category=dpg.mvThemeCat_Plots,
                    )
                    dpg.add_theme_style(
                        dpg.mvPlotStyleVar_LineWeight,
                        1.0 if role == "grid_minor" else 1.8,
                        category=dpg.mvThemeCat_Plots,
                    )
        with dpg.theme(tag=f"medical.role.{name}.danger_button"):
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (183, 45, 55, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (211, 57, 69, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (150, 35, 44, 255))

    def apply(self, button_tag: str | None = None) -> None:
        mode = "dark" if self.dark else "light"
        self.dpg.bind_theme(f"medical.theme.{mode}")
        for item, role in self._bindings:
            if self.dpg.does_item_exist(item):
                self.dpg.bind_item_theme(item, f"medical.role.{mode}.{role}")
        if button_tag and self.dpg.does_item_exist(button_tag):
            self.dpg.set_item_label(button_tag, "Light mode" if self.dark else "Dark mode")

    def toggle(self, button_tag: str | None = None) -> None:
        self.dark = not self.dark
        self.apply(button_tag)

    def bind_role(self, item: str, role: str) -> None:
        self._bindings.append((item, role))
        mode = "dark" if self.dark else "light"
        if self.dpg.does_item_exist(item):
            self.dpg.bind_item_theme(item, f"medical.role.{mode}.{role}")

    def bind_heading(self, item: str, metric: bool = False) -> None:
        font = "medical.font.metric" if metric else "medical.font.heading"
        if self.dpg.does_item_exist(font) and self.dpg.does_item_exist(item):
            self.dpg.bind_item_font(item, font)


def configure_global_theme(dpg, dark: bool = True) -> MedicalThemeController:
    return MedicalThemeController(dpg, dark=dark)


def replace_table_rows(dpg, table_tag: str, rows: Iterable[Iterable[object]]) -> None:
    dpg.delete_item(table_tag, children_only=True, slot=1)
    for row in rows:
        with dpg.table_row(parent=table_tag):
            for value in row:
                dpg.add_text(str(value))
