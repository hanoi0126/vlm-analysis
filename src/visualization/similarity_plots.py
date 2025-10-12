"""Visualization for representation similarity analysis.

This module provides plotting functions for visualizing similarity metrics
and 2D projections of representation spaces.
"""

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing_extensions import assert_never
import umap

from src.visualization.plots import TASK_COLORS


def plot_similarity_curves(
    layers: list[str],
    similarities: dict[str, dict[str, float]],
    task: str,
    metrics: list[str] | None = None,
    title_suffix: str = "",
    figsize: tuple[float, float] = (14, 6),
    output_path: str | Path | None = None,
) -> None:
    """Plot similarity metrics across layers.

    Creates subplots showing how similarity metrics (CKA, cosine, etc.)
    vary across layers.

    Args:
        layers: List of layer names in order
        similarities: Dictionary mapping layer name to metrics dict
            Example: {"l00": {"cka": 0.95, "cosine": 0.87}, ...}
        task: Task name (used for coloring)
        metrics: List of metrics to plot (e.g., ["cka", "cosine", "procrustes"])
            If None, defaults to ["cka", "cosine"]
        title_suffix: Additional text to append to plot titles
        figsize: Figure size as (width, height)
        output_path: Path to save the plot. If None, only displays the plot
    """
    if metrics is None:
        metrics = ["cka", "cosine"]

    fig, axes = plt.subplots(len(metrics), 1, figsize=figsize, squeeze=False)

    x = np.arange(len(layers))
    color = TASK_COLORS.get(task, "tab:blue")

    for idx, metric in enumerate(metrics):
        ax = axes[idx, 0]

        # Extract metric values
        values = np.array([similarities[layer].get(metric, np.nan) if layer in similarities else np.nan for layer in layers])

        # Plot
        ax.plot(x, values, marker="o", linestyle="-", color=color, linewidth=2, markersize=6)

        # Format x-axis
        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=90, fontsize=8)
        ax.grid(visible=True, linestyle="--", alpha=0.6)

        # Set labels and title
        metric_name = metric.upper() if metric != "procrustes" else "Procrustes Distance"
        title = f"{task} — {metric_name}"
        if title_suffix:
            title += f" ({title_suffix})"
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Layer", fontsize=10)
        ax.set_ylabel(metric_name, fontsize=10)

        # Add horizontal reference lines
        if metric in ["cka", "cosine"]:
            ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5, linewidth=1)
            ax.axhline(0.5, color="gray", linestyle=":", alpha=0.3, linewidth=1)
            ax.set_ylim(-0.1, 1.1)

    fig.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved plot to: {output_path}")

    plt.show()
    plt.close(fig)


def plot_2d_comparison(
    features_a: NDArray[np.float64],
    features_b: NDArray[np.float64],
    labels: NDArray[np.int64],
    layer: str,
    task: str,
    method: Literal["pca", "tsne", "umap"] = "pca",
    figsize: tuple[float, float] = (18, 6),
    output_path: str | Path | None = None,
) -> None:
    """Plot 2D comparison of Vision and Text representations.

    Creates three subplots:
    1. Vision space (Image ON) with class labels
    2. Text space (Image OFF) with class labels
    3. Overlay showing Vision → Text transformation with arrows

    Args:
        features_a: Vision features of shape (N, D)
        features_b: Text features of shape (N, D)
        labels: Class labels of shape (N,)
        layer: Layer name
        task: Task name (used for titles)
        method: Dimensionality reduction method ("pca", "tsne", or "umap")
        figsize: Figure size as (width, height)
        output_path: Path to save the plot. If None, only displays the plot
    """
    # Dimensionality reduction
    if method == "pca":
        reducer = PCA(n_components=2, random_state=0)
    elif method == "tsne":
        reducer = TSNE(n_components=2, random_state=0, perplexity=min(30, len(labels) - 1))
    elif method == "umap":
        reducer = umap.UMAP(n_components=2, random_state=0)
    else:
        assert_never(method)

    # Fit on combined data for consistent projection
    combined = np.vstack([features_a, features_b])
    combined_2d = reducer.fit_transform(combined)

    features_a_2d = combined_2d[: len(features_a)]
    features_b_2d = combined_2d[len(features_a) :]

    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    # Get unique labels and colors
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    cmap = plt.cm.get_cmap("tab10" if n_classes <= 10 else "tab20")

    # Plot A (Vision)
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax1.scatter(
            features_a_2d[mask, 0],
            features_a_2d[mask, 1],
            c=[cmap(i)],
            label=f"Class {label}",
            alpha=0.6,
            s=50,
            edgecolors="white",
            linewidth=0.5,
        )
    ax1.set_title(f"Image ON (Vision)\n{task} — {layer}", fontweight="bold", fontsize=11)
    ax1.set_xlabel(f"{method.upper()} 1", fontsize=10)
    ax1.set_ylabel(f"{method.upper()} 2", fontsize=10)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot B (Text)
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax2.scatter(
            features_b_2d[mask, 0],
            features_b_2d[mask, 1],
            c=[cmap(i)],
            label=f"Class {label}",
            alpha=0.6,
            s=50,
            edgecolors="white",
            linewidth=0.5,
        )
    ax2.set_title(f"Image OFF (Text)\n{task} — {layer}", fontweight="bold", fontsize=11)
    ax2.set_xlabel(f"{method.upper()} 1", fontsize=10)
    ax2.set_ylabel(f"{method.upper()} 2", fontsize=10)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot overlay with arrows
    for i, label in enumerate(unique_labels):
        mask = labels == label
        # Plot both conditions
        ax3.scatter(
            features_a_2d[mask, 0],
            features_a_2d[mask, 1],
            c=[cmap(i)],
            marker="o",
            alpha=0.4,
            s=50,
            label=f"Class {label} (Vision)",
            edgecolors="white",
            linewidth=0.5,
        )
        ax3.scatter(
            features_b_2d[mask, 0],
            features_b_2d[mask, 1],
            c=[cmap(i)],
            marker="s",
            alpha=0.4,
            s=50,
            edgecolors="white",
            linewidth=0.5,
        )

        # Draw arrows showing transformation
        # Sample a few points to avoid clutter
        n_samples = min(10, int(mask.sum()))
        if n_samples > 0:
            indices = np.where(mask)[0]
            sample_indices = np.random.choice(indices, size=n_samples, replace=False)
            for idx in sample_indices:
                dx = features_b_2d[idx, 0] - features_a_2d[idx, 0]
                dy = features_b_2d[idx, 1] - features_a_2d[idx, 1]
                arrow_length = np.sqrt(dx**2 + dy**2)
                if arrow_length > 0.01:  # Only draw if movement is visible
                    ax3.arrow(
                        features_a_2d[idx, 0],
                        features_a_2d[idx, 1],
                        dx,
                        dy,
                        color=cmap(i),
                        alpha=0.3,
                        width=0.01,
                        head_width=0.1,
                        length_includes_head=True,
                    )

    ax3.set_title(f"Vision → Text Transformation\n{task} — {layer}", fontweight="bold", fontsize=11)
    ax3.set_xlabel(f"{method.upper()} 1", fontsize=10)
    ax3.set_ylabel(f"{method.upper()} 2", fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Add legend (only unique class labels, not both markers)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap(i), markersize=8, label=f"Class {label}")
        for i, label in enumerate(unique_labels)
    ]
    ax3.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    # Unify axis scales across all three subplots
    all_x = np.concatenate([features_a_2d[:, 0], features_b_2d[:, 0]])
    all_y = np.concatenate([features_a_2d[:, 1], features_b_2d[:, 1]])
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1

    ax1.set_xlim(x_min - x_margin, x_max + x_margin)
    ax1.set_ylim(y_min - y_margin, y_max + y_margin)
    ax2.set_xlim(x_min - x_margin, x_max + x_margin)
    ax2.set_ylim(y_min - y_margin, y_max + y_margin)
    ax3.set_xlim(x_min - x_margin, x_max + x_margin)
    ax3.set_ylim(y_min - y_margin, y_max + y_margin)

    fig.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved plot to: {output_path}")

    plt.show()
    plt.close(fig)


def plot_layer_trajectory(
    all_features_a: dict[str, NDArray[np.float64]],
    all_features_b: dict[str, NDArray[np.float64]],
    labels: NDArray[np.int64],
    selected_layers: list[str],
    task: str,
    method: Literal["pca", "tsne", "umap"] = "pca",
    sample_size: int = 100,
    n_arrows_per_class: int = 3,
    title_suffix: str = "",
    figsize: tuple[float, float] = (24, 7),
    output_path: str | Path | None = None,
) -> None:
    """Plot trajectory of representations across layers in 2D space.

    Creates 3 panels to visualize how representations evolve through layers:
    1. Vision space trajectory (Image ON)
    2. Text space trajectory (Image OFF)
    3. Overlay comparison (both trajectories overlaid)

    Layer progression is indicated by:
    - Size: Early layers = small, later layers = large
    - Edge color: Blue (early) → Red (later)
    - Layer labels at starting positions

    Args:
        all_features_a: Dict mapping layer names to Vision features of shape (N, D)
        all_features_b: Dict mapping layer names to Text features of shape (N, D)
        labels: Class labels of shape (N,)
        selected_layers: List of layer names to visualize (in order)
        task: Task name (used for titles)
        method: Dimensionality reduction method ("pca", "tsne", or "umap")
        sample_size: Maximum number of samples to plot (for clarity)
        n_arrows_per_class: Number of trajectory arrows to draw per class
        title_suffix: Additional text to append to plot titles
        figsize: Figure size as (width, height)
        output_path: Path to save the plot. If None, only displays the plot
    """
    # Sample data for clarity
    n = len(labels)
    if n > sample_size:
        indices = np.random.choice(n, size=sample_size, replace=False)
    else:
        indices = np.arange(n)

    sampled_labels = labels[indices]

    # Prepare data for each layer
    trajectories_a: list[NDArray[np.float64]] = []
    trajectories_b: list[NDArray[np.float64]] = []
    valid_layers: list[str] = []

    for layer in selected_layers:
        if layer in all_features_a and layer in all_features_b:
            features_a = all_features_a[layer][indices]
            features_b = all_features_b[layer][indices]

            # Reduce to 2D
            if method == "pca":
                reducer = PCA(n_components=2, random_state=0)
            elif method == "tsne":
                perplexity = min(30, len(indices) - 1)
                reducer = TSNE(n_components=2, random_state=0, perplexity=perplexity)
            elif method == "umap":
                reducer = umap.UMAP(n_components=2, random_state=0)
            else:
                assert_never(method)

            # Fit on combined for consistency
            combined = np.vstack([features_a, features_b])
            combined_2d = reducer.fit_transform(combined)

            features_a_2d = combined_2d[: len(features_a)]
            features_b_2d = combined_2d[len(features_a) :]

            trajectories_a.append(features_a_2d)
            trajectories_b.append(features_b_2d)
            valid_layers.append(layer)

    if not trajectories_a:
        print("[WARN] No valid layers found for trajectory plot")
        return

    # Create 3-panel figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    unique_labels = np.unique(sampled_labels)
    n_classes = len(unique_labels)
    class_cmap = plt.cm.get_cmap("tab10" if n_classes <= 10 else "tab20")

    # Layer colormap (blue to red for progression)
    layer_cmap = plt.cm.get_cmap("coolwarm")
    n_layers = len(valid_layers)

    # Calculate common scale for all panels
    all_points_a = np.vstack(trajectories_a)
    all_points_b = np.vstack(trajectories_b)
    all_points = np.vstack([all_points_a, all_points_b])

    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()

    x_range = x_max - x_min
    y_range = y_max - y_min
    padding = 0.15

    x_min -= x_range * padding
    x_max += x_range * padding
    y_min -= y_range * padding
    y_max += y_range * padding

    # =========================================================================
    # Panel 1: Vision Trajectory
    # =========================================================================
    for i, label in enumerate(unique_labels):
        mask = sampled_labels == label

        # Plot points for each layer with varying size and color
        for j, layer in enumerate(valid_layers):
            points = trajectories_a[j][mask]

            # Size increases with layer depth
            base_size = 40
            size = base_size + 100 * (j / max(1, n_layers - 1))

            # Color transitions from blue to red
            layer_color = layer_cmap(j / max(1, n_layers - 1))

            ax1.scatter(
                points[:, 0],
                points[:, 1],
                c=[class_cmap(i)] * len(points),
                s=size,
                alpha=0.7,
                edgecolors=layer_color,
                linewidth=2.5,
                label=f"Class {label}" if j == 0 else "",
                zorder=10 + j,
            )

            # Add layer label at the centroid of first layer
            if j == 0 and mask.sum() > 0:
                centroid = points.mean(axis=0)
                ax1.text(
                    centroid[0],
                    centroid[1],
                    layer,
                    fontsize=9,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    bbox={
                        "boxstyle": "round,pad=0.3",
                        "facecolor": "white",
                        "alpha": 0.8,
                        "edgecolor": layer_color,
                        "linewidth": 1.5,
                    },
                    zorder=50,
                )

        # Draw arrows
        if mask.sum() > 0:
            n_arrows = min(n_arrows_per_class, int(mask.sum()))
            arrow_indices = np.where(mask)[0]
            if n_arrows < mask.sum():
                arrow_indices = np.random.choice(arrow_indices, size=n_arrows, replace=False)

            for idx in arrow_indices:
                for j in range(1, len(valid_layers)):
                    start = trajectories_a[j - 1][idx]
                    end = trajectories_a[j][idx]

                    dx = end[0] - start[0]
                    dy = end[1] - start[1]
                    length = np.sqrt(dx**2 + dy**2)

                    if length > 0.5:  # Only draw if movement is significant
                        layer_alpha = 0.3 + 0.4 * (j / n_layers)
                        ax1.annotate(
                            "",
                            xy=(end[0], end[1]),
                            xytext=(start[0], start[1]),
                            arrowprops={
                                "arrowstyle": "-|>",
                                "lw": 2.5,
                                "color": class_cmap(i),
                                "alpha": layer_alpha,
                                "mutation_scale": 25,
                                "shrinkA": 0,
                                "shrinkB": 0,
                            },
                            zorder=5,
                        )

    title = f"Vision Trajectory (Image ON)\n{task}"
    if title_suffix:
        title += f" ({title_suffix})"
    ax1.set_title(title, fontsize=15, fontweight="bold", pad=15)
    ax1.set_xlabel(f"{method.upper()} Component 1", fontsize=12)
    ax1.set_ylabel(f"{method.upper()} Component 2", fontsize=12)
    ax1.legend(loc="best", framealpha=0.95, fontsize=10, title="Class", title_fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_aspect("equal", adjustable="box")

    # Add layer progression indicator
    ax1.text(
        0.02,
        0.02,
        f"Layers: {valid_layers[0]} → {valid_layers[-1]}",
        transform=ax1.transAxes,
        verticalalignment="bottom",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.7},
        fontsize=9,
    )

    # =========================================================================
    # Panel 2: Text Trajectory
    # =========================================================================
    for i, label in enumerate(unique_labels):
        mask = sampled_labels == label

        for j, layer in enumerate(valid_layers):
            points = trajectories_b[j][mask]

            base_size = 40
            size = base_size + 100 * (j / max(1, n_layers - 1))
            layer_color = layer_cmap(j / max(1, n_layers - 1))

            ax2.scatter(
                points[:, 0],
                points[:, 1],
                c=[class_cmap(i)] * len(points),
                s=size,
                marker="s",
                alpha=0.7,
                edgecolors=layer_color,
                linewidth=2.5,
                label=f"Class {label}" if j == 0 else "",
                zorder=10 + j,
            )

            # Add layer label
            if j == 0 and mask.sum() > 0:
                centroid = points.mean(axis=0)
                ax2.text(
                    centroid[0],
                    centroid[1],
                    layer,
                    fontsize=9,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    bbox={
                        "boxstyle": "round,pad=0.3",
                        "facecolor": "white",
                        "alpha": 0.8,
                        "edgecolor": layer_color,
                        "linewidth": 1.5,
                    },
                    zorder=50,
                )

        # Draw arrows
        if mask.sum() > 0:
            n_arrows = min(n_arrows_per_class, int(mask.sum()))
            arrow_indices = np.where(mask)[0]
            if n_arrows < mask.sum():
                arrow_indices = np.random.choice(arrow_indices, size=n_arrows, replace=False)

            for idx in arrow_indices:
                for j in range(1, len(valid_layers)):
                    start = trajectories_b[j - 1][idx]
                    end = trajectories_b[j][idx]

                    dx = end[0] - start[0]
                    dy = end[1] - start[1]
                    length = np.sqrt(dx**2 + dy**2)

                    if length > 0.5:
                        layer_alpha = 0.3 + 0.4 * (j / n_layers)
                        ax2.annotate(
                            "",
                            xy=(end[0], end[1]),
                            xytext=(start[0], start[1]),
                            arrowprops={
                                "arrowstyle": "-|>",
                                "lw": 2.5,
                                "color": class_cmap(i),
                                "alpha": layer_alpha,
                                "mutation_scale": 25,
                                "shrinkA": 0,
                                "shrinkB": 0,
                            },
                            zorder=5,
                        )

    title = f"Text Trajectory (Image OFF)\n{task}"
    if title_suffix:
        title += f" ({title_suffix})"
    ax2.set_title(title, fontsize=15, fontweight="bold", pad=15)
    ax2.set_xlabel(f"{method.upper()} Component 1", fontsize=12)
    ax2.set_ylabel(f"{method.upper()} Component 2", fontsize=12)
    ax2.legend(loc="best", framealpha=0.95, fontsize=10, title="Class", title_fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_aspect("equal", adjustable="box")

    ax2.text(
        0.02,
        0.02,
        f"Layers: {valid_layers[0]} → {valid_layers[-1]}",
        transform=ax2.transAxes,
        verticalalignment="bottom",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.7},
        fontsize=9,
    )

    # =========================================================================
    # Panel 3: Overlay Comparison
    # =========================================================================
    for i, label in enumerate(unique_labels):
        mask = sampled_labels == label

        # Plot both trajectories on same axes
        for j, layer in enumerate(valid_layers):
            points_a = trajectories_a[j][mask]
            points_b = trajectories_b[j][mask]

            base_size = 40
            size = base_size + 100 * (j / max(1, n_layers - 1))
            layer_color = layer_cmap(j / max(1, n_layers - 1))

            # Vision (circles)
            ax3.scatter(
                points_a[:, 0],
                points_a[:, 1],
                c=[class_cmap(i)] * len(points_a),
                s=size,
                alpha=0.6,
                edgecolors=layer_color,
                linewidth=2.0,
                label=f"Class {label} (Vision)" if j == 0 else "",
                zorder=10 + j,
            )

            # Text (squares)
            ax3.scatter(
                points_b[:, 0],
                points_b[:, 1],
                c=[class_cmap(i)] * len(points_b),
                s=size,
                marker="s",
                alpha=0.6,
                edgecolors=layer_color,
                linewidth=2.0,
                label=f"Class {label} (Text)" if j == 0 else "",
                zorder=10 + j,
            )

            # Add layer labels for first layer only
            if j == 0 and mask.sum() > 0:
                # Vision label
                centroid_a = points_a.mean(axis=0)
                ax3.text(
                    centroid_a[0],
                    centroid_a[1] + 2,
                    f"{layer}\n(V)",
                    fontsize=8,
                    fontweight="bold",
                    ha="center",
                    va="bottom",
                    bbox={
                        "boxstyle": "round,pad=0.2",
                        "facecolor": "lightblue",
                        "alpha": 0.7,
                        "edgecolor": layer_color,
                        "linewidth": 1.0,
                    },
                    zorder=50,
                )

                # Text label
                centroid_b = points_b.mean(axis=0)
                ax3.text(
                    centroid_b[0],
                    centroid_b[1] - 2,
                    f"{layer}\n(T)",
                    fontsize=8,
                    fontweight="bold",
                    ha="center",
                    va="top",
                    bbox={
                        "boxstyle": "round,pad=0.2",
                        "facecolor": "lightgreen",
                        "alpha": 0.7,
                        "edgecolor": layer_color,
                        "linewidth": 1.0,
                    },
                    zorder=50,
                )

        # Draw arrows
        if mask.sum() > 0:
            n_arrows = min(n_arrows_per_class, int(mask.sum()))
            arrow_indices = np.where(mask)[0]
            if n_arrows < mask.sum():
                arrow_indices = np.random.choice(arrow_indices, size=n_arrows, replace=False)

            for idx in arrow_indices:
                # Vision arrows (solid)
                for j in range(1, len(valid_layers)):
                    start = trajectories_a[j - 1][idx]
                    end = trajectories_a[j][idx]

                    dx = end[0] - start[0]
                    dy = end[1] - start[1]
                    length = np.sqrt(dx**2 + dy**2)

                    if length > 0.5:
                        layer_alpha = 0.3 + 0.3 * (j / n_layers)
                        ax3.annotate(
                            "",
                            xy=(end[0], end[1]),
                            xytext=(start[0], start[1]),
                            arrowprops={
                                "arrowstyle": "-|>",
                                "lw": 2.0,
                                "color": class_cmap(i),
                                "alpha": layer_alpha,
                                "mutation_scale": 20,
                                "linestyle": "-",
                                "shrinkA": 0,
                                "shrinkB": 0,
                            },
                            zorder=5,
                        )

                # Text arrows (dashed)
                for j in range(1, len(valid_layers)):
                    start = trajectories_b[j - 1][idx]
                    end = trajectories_b[j][idx]

                    dx = end[0] - start[0]
                    dy = end[1] - start[1]
                    length = np.sqrt(dx**2 + dy**2)

                    if length > 0.5:
                        layer_alpha = 0.3 + 0.3 * (j / n_layers)
                        ax3.annotate(
                            "",
                            xy=(end[0], end[1]),
                            xytext=(start[0], start[1]),
                            arrowprops={
                                "arrowstyle": "-|>",
                                "lw": 2.0,
                                "color": class_cmap(i),
                                "alpha": layer_alpha,
                                "mutation_scale": 20,
                                "linestyle": "--",
                                "shrinkA": 0,
                                "shrinkB": 0,
                            },
                            zorder=5,
                        )

    title = f"Overlay: Vision vs Text\n{task}"
    if title_suffix:
        title += f" ({title_suffix})"
    ax3.set_title(title, fontsize=15, fontweight="bold", pad=15)
    ax3.set_xlabel(f"{method.upper()} Component 1", fontsize=12)
    ax3.set_ylabel(f"{method.upper()} Component 2", fontsize=12)

    # Custom legend (avoid too many entries)
    handles, labels_legend = ax3.get_legend_handles_labels()
    # Keep only unique class labels
    by_label: dict[str, plt.Artist] = {}
    for h, label_text in zip(handles, labels_legend, strict=True):
        base_label = label_text.split(" (")[0]  # Remove (Vision)/(Text) suffix
        if base_label not in by_label:
            by_label[base_label] = h

    ax3.legend(
        by_label.values(),
        by_label.keys(),
        loc="best",
        framealpha=0.95,
        fontsize=10,
        title="Class (○=Vision, □=Text)",
        title_fontsize=10,
    )
    ax3.grid(True, alpha=0.3, linestyle="--")
    ax3.set_xlim(x_min, x_max)
    ax3.set_ylim(y_min, y_max)
    ax3.set_aspect("equal", adjustable="box")

    # Add annotation
    ax3.text(
        0.02,
        0.02,
        f"Layers: {valid_layers[0]} → {valid_layers[-1]}\nSolid=Vision, Dashed=Text",
        transform=ax3.transAxes,
        verticalalignment="bottom",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.7},
        fontsize=9,
    )

    fig.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved plot to: {output_path}")

    plt.show()
    plt.close(fig)
