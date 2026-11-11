from __future__ import annotations

from pathlib import Path
from typing import Iterable


def choose_directory(title: str, initial: str | Path | None = None) -> str:
    return _run_dialog(
        "askdirectory",
        title=title,
        initialdir=_initial_directory(initial),
        mustexist=True,
    )


def choose_open_file(
    title: str,
    filetypes: Iterable[tuple[str, str]],
    initial: str | Path | None = None,
) -> str:
    return _run_dialog(
        "askopenfilename",
        title=title,
        initialdir=_initial_directory(initial),
        filetypes=list(filetypes),
    )


def choose_save_file(
    title: str,
    filetypes: Iterable[tuple[str, str]],
    default_extension: str,
    initial: str | Path | None = None,
    initial_file: str = "",
) -> str:
    return _run_dialog(
        "asksaveasfilename",
        title=title,
        initialdir=_initial_directory(initial),
        initialfile=initial_file,
        filetypes=list(filetypes),
        defaultextension=default_extension,
    )


def _run_dialog(method: str, **kwargs) -> str:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update_idletasks()
    try:
        value = getattr(filedialog, method)(parent=root, **kwargs)
        return str(value or "")
    finally:
        root.destroy()


def _initial_directory(value: str | Path | None) -> str:
    if value:
        path = Path(value).expanduser()
        if path.is_file():
            path = path.parent
        if path.is_dir():
            return str(path)
    return str(Path.cwd())
