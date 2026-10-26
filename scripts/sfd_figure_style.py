"""Shared typography defaults for SFD manuscript figures."""

from __future__ import annotations

from typing import Any

SFD_FONT_FAMILY = "Times New Roman"
SFD_FONT_FALLBACKS = ["Times New Roman", "Times", "DejaVu Serif"]
SFD_INK = "#243447"
SFD_MUTED = "#5E6975"
SFD_GRID = "#DDE6EE"
PANEL_TITLE_SIZE = 10.0
PANEL_LABEL_SIZE = 10.0
SUBTITLE_SIZE = 9.0
AXIS_LABEL_SIZE = 9.0
TICK_LABEL_SIZE = 8.0


def set_sfd_matplotlib_style(
    plt: Any,
    *,
    base_size: float = 9.0,
    title_size: float = PANEL_TITLE_SIZE,
    label_size: float = AXIS_LABEL_SIZE,
    tick_size: float = TICK_LABEL_SIZE,
) -> None:
    """Apply manuscript-facing Matplotlib typography and PDF font settings."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": SFD_FONT_FALLBACKS,
            "font.size": base_size,
            "axes.titlesize": title_size,
            "axes.labelsize": label_size,
            "axes.titleweight": "bold",
            "axes.labelcolor": SFD_INK,
            "axes.edgecolor": SFD_INK,
            "axes.linewidth": 0.8,
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
            "xtick.color": SFD_INK,
            "ytick.color": SFD_INK,
            "text.color": SFD_INK,
            "mathtext.fontset": "custom",
            "mathtext.rm": SFD_FONT_FAMILY,
            "mathtext.it": f"{SFD_FONT_FAMILY}:italic",
            "mathtext.bf": f"{SFD_FONT_FAMILY}:bold",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_title(ax: Any, text: str, *, pad: float | None = None, size: float = PANEL_TITLE_SIZE) -> None:
    """Set a left-aligned SFD panel title with the shared title hierarchy."""

    kwargs: dict[str, Any] = {
        "loc": "left",
        "fontsize": size,
        "fontweight": "bold",
        "fontfamily": SFD_FONT_FAMILY,
        "color": SFD_INK,
    }
    if pad is not None:
        kwargs["pad"] = pad
    ax.set_title(text, **kwargs)


def panel_label(
    ax: Any,
    label: str,
    *,
    x: float = -0.10,
    y: float = 1.05,
    size: float = PANEL_LABEL_SIZE,
) -> None:
    """Draw a standalone panel label using the shared label hierarchy."""

    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=size,
        fontweight="bold",
        fontfamily=SFD_FONT_FAMILY,
        color=SFD_INK,
        va="bottom",
        ha="left",
    )
