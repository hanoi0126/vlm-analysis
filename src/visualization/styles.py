"""Unified visual styles for Vision/Text comparison plots."""

from typing import Any

import matplotlib.pyplot as plt

# ============================================================================
# General Plot Style
# ============================================================================


def apply_plot_style() -> None:
    """
    Apply general plot style for consistency across all visualizations.

    Sets matplotlib style parameters for clean, publication-ready plots.
    """
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )


# ============================================================================
# Vision/Text Style Definitions
# ============================================================================

VISION_STYLE = {
    "marker": "o",
    "markersize": 8,
    "alpha": 0.85,
    "edgecolor": "black",
    "linewidth": 2.5,
    "linestyle": "-",
    "label_prefix": "●",
}

TEXT_STYLE = {
    "marker": "s",
    "markersize": 8,
    "alpha": 0.65,
    "edgecolor": "white",
    "linewidth": 2.5,
    "linestyle": "--",
    "label_prefix": "□",
}


# ============================================================================
# Color Palettes
# ============================================================================

# Rich, saturated colors for Vision
VISION_COLORS = [
    "#1f77b4",  # Rich blue
    "#ff7f0e",  # Rich orange
    "#2ca02c",  # Rich green
    "#d62728",  # Rich red
    "#9467bd",  # Rich purple
    "#8c564b",  # Rich brown
    "#e377c2",  # Rich pink
    "#7f7f7f",  # Rich gray
    "#bcbd22",  # Rich olive
    "#17becf",  # Rich cyan
]

# Desaturated, lighter colors for Text
TEXT_COLORS = [
    "#aec7e8",  # Light blue
    "#ffbb78",  # Light orange
    "#98df8a",  # Light green
    "#ff9896",  # Light red
    "#c5b0d5",  # Light purple
    "#c49c94",  # Light brown
    "#f7b6d2",  # Light pink
    "#c7c7c7",  # Light gray
    "#dbdb8d",  # Light olive
    "#9edae5",  # Light cyan
]


# ============================================================================
# Helper Functions
# ============================================================================


def get_vision_color(class_idx: int, n_classes: int = 10) -> str:
    """
    Get color for Vision representation.

    Args:
        class_idx: Class index
        n_classes: Total number of classes

    Returns:
        Hex color string
    """
    if n_classes <= 10:
        return VISION_COLORS[class_idx % len(VISION_COLORS)]
    # Use matplotlib colormap for many classes
    cmap = plt.cm.get_cmap("tab20")
    color = cmap(class_idx % 20)
    # Convert to hex
    return f"#{int(color[0] * 255):02x}{int(color[1] * 255):02x}{int(color[2] * 255):02x}"


def get_text_color(class_idx: int, n_classes: int = 10) -> str:
    """
    Get color for Text representation.

    Args:
        class_idx: Class index
        n_classes: Total number of classes

    Returns:
        Hex color string
    """
    if n_classes <= 10:
        return TEXT_COLORS[class_idx % len(TEXT_COLORS)]
    # Use matplotlib colormap for many classes
    cmap = plt.cm.get_cmap("Pastel1")
    color = cmap(class_idx % 9)
    # Convert to hex
    return f"#{int(color[0] * 255):02x}{int(color[1] * 255):02x}{int(color[2] * 255):02x}"


def apply_vision_style(**kwargs: Any) -> dict[str, Any]:
    """
    Get scatter/plot kwargs with Vision style applied.

    Args:
        **kwargs: Additional style overrides

    Returns:
        Complete style dictionary
    """
    style = VISION_STYLE.copy()
    style.update(kwargs)
    return style


def apply_text_style(**kwargs: Any) -> dict[str, Any]:
    """
    Get scatter/plot kwargs with Text style applied.

    Args:
        **kwargs: Additional style overrides

    Returns:
        Complete style dictionary
    """
    style = TEXT_STYLE.copy()
    style.update(kwargs)
    return style


def create_unified_legend(
    ax: plt.Axes,
    class_names: list[str] | None = None,
    include_both: bool = True,
    **legend_kwargs: Any,
) -> None:
    """
    Create unified legend showing Vision/Text distinction.

    Args:
        ax: Matplotlib axes
        class_names: List of class names (optional)
        include_both: Include both Vision and Text in legend
        **legend_kwargs: Additional legend arguments
    """
    handles = []
    labels = []

    if class_names is None:
        class_names = [f"Class {i}" for i in range(10)]

    for i, name in enumerate(class_names):
        if include_both:
            # Vision marker
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker=str(VISION_STYLE["marker"]),
                    color="w",
                    markerfacecolor=get_vision_color(i, len(class_names)),
                    markersize=10,
                    markeredgecolor=str(VISION_STYLE["edgecolor"]),
                    markeredgewidth=2,
                    linestyle="none",
                )
            )
            labels.append(f"{VISION_STYLE['label_prefix']} {name} (Vision)")

            # Text marker
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker=str(TEXT_STYLE["marker"]),
                    color="w",
                    markerfacecolor=get_text_color(i, len(class_names)),
                    markersize=10,
                    markeredgecolor=str(TEXT_STYLE["edgecolor"]),
                    markeredgewidth=2,
                    linestyle="none",
                )
            )
            labels.append(f"{TEXT_STYLE['label_prefix']} {name} (Text)")
        else:
            # Single entry (for single-condition plots)
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=get_vision_color(i, len(class_names)),
                    markersize=10,
                    markeredgecolor="black",
                    markeredgewidth=2,
                    linestyle="none",
                )
            )
            labels.append(name)

    default_kwargs = {
        "loc": "best",
        "framealpha": 0.95,
        "fontsize": 10,
        "ncol": 1,
    }
    default_kwargs.update(legend_kwargs)

    ax.legend(handles, labels, **default_kwargs)


def add_style_annotation(
    ax: plt.Axes,
    position: str = "bottom_right",
) -> None:
    """
    Add annotation explaining Vision/Text visual coding.

    Args:
        ax: Matplotlib axes
        position: Where to place annotation ('bottom_right', 'top_left', etc.)
    """
    text = (
        f"{VISION_STYLE['label_prefix']} Vision (Image ON): "
        f"{VISION_STYLE['marker']} darker, {VISION_STYLE['linestyle']}\n"
        f"{TEXT_STYLE['label_prefix']} Text (Image OFF): "
        f"{TEXT_STYLE['marker']} lighter, {TEXT_STYLE['linestyle']}"
    )

    positions = {
        "bottom_right": (0.98, 0.02),
        "bottom_left": (0.02, 0.02),
        "top_right": (0.98, 0.98),
        "top_left": (0.02, 0.98),
    }

    va_map = {
        "bottom_right": "bottom",
        "bottom_left": "bottom",
        "top_right": "top",
        "top_left": "top",
    }

    ha_map = {
        "bottom_right": "right",
        "bottom_left": "left",
        "top_right": "right",
        "top_left": "left",
    }

    x, y = positions.get(position, (0.98, 0.02))
    va = va_map.get(position, "bottom")
    ha = ha_map.get(position, "right")

    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment=va,
        horizontalalignment=ha,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "wheat",
            "alpha": 0.8,
            "edgecolor": "black",
            "linewidth": 1.5,
        },
        zorder=100,
    )


# ============================================================================
# Preset Configurations
# ============================================================================


def get_scatter_kwargs(
    condition: str,  # 'vision' or 'text'
    class_idx: int,
    n_classes: int = 10,
    size: float = 100,
    **overrides: Any,
) -> dict[str, Any]:
    """
    Get complete scatter plot kwargs with unified styling.

    Args:
        condition: 'vision' or 'text'
        class_idx: Class index for color
        n_classes: Total number of classes
        size: Marker size
        **overrides: Style overrides

    Returns:
        Complete kwargs dictionary for ax.scatter()
    """
    if condition.lower() == "vision":
        style = VISION_STYLE.copy()
        color = get_vision_color(class_idx, n_classes)
    elif condition.lower() == "text":
        style = TEXT_STYLE.copy()
        color = get_text_color(class_idx, n_classes)
    else:
        msg = f"Unknown condition: {condition}"
        raise ValueError(msg)

    kwargs = {
        "c": color,
        "s": size,
        "marker": style["marker"],
        "alpha": style["alpha"],
        "edgecolors": style["edgecolor"],
        "linewidth": style["linewidth"],
        "zorder": 10,
    }

    kwargs.update(overrides)
    return kwargs


def get_line_kwargs(
    condition: str,  # 'vision' or 'text'
    class_idx: int,
    n_classes: int = 10,
    linewidth: float = 3.0,
    **overrides: Any,
) -> dict[str, Any]:
    """
    Get complete line plot kwargs with unified styling.

    Args:
        condition: 'vision' or 'text'
        class_idx: Class index for color
        n_classes: Total number of classes
        linewidth: Line width
        **overrides: Style overrides

    Returns:
        Complete kwargs dictionary for ax.plot()
    """
    if condition.lower() == "vision":
        style = VISION_STYLE.copy()
        color = get_vision_color(class_idx, n_classes)
    elif condition.lower() == "text":
        style = TEXT_STYLE.copy()
        color = get_text_color(class_idx, n_classes)
    else:
        msg = f"Unknown condition: {condition}"
        raise ValueError(msg)

    kwargs = {
        "color": color,
        "linewidth": linewidth,
        "linestyle": style["linestyle"],
        "alpha": style["alpha"],
        "marker": style["marker"],
        "markersize": style["markersize"],
        "markeredgecolor": style["edgecolor"],
        "markeredgewidth": style["linewidth"],
        "zorder": 10,
    }

    kwargs.update(overrides)
    return kwargs
