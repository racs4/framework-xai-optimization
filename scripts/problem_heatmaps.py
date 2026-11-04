import base64
import re
import zlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Rectangle


def cell_to_array(cell, shape, dtype=np.float32):
    compressed = base64.b64decode(cell)
    raw = zlib.decompress(compressed)
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


def _sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_")


def _decode_heatmap(cell_value: str, shape_1: int, shape_2: int) -> np.ndarray:
    heatmap = cell_to_array(cell_value, (shape_1, shape_2))
    return np.asarray(heatmap, dtype=np.float32)


def _center_place(array_2d: np.ndarray, target_h: int, target_w: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = array_2d.shape

    if h > target_h or w > target_w:
        raise ValueError("Array shape is larger than target canvas.")

    top = (target_h - h) // 2
    left = (target_w - w) // 2

    canvas = np.zeros((target_h, target_w), dtype=np.float32)
    valid = np.zeros((target_h, target_w), dtype=np.float32)

    canvas[top:top + h, left:left + w] = array_2d
    valid[top:top + h, left:left + w] = 1.0

    return canvas, valid


def generate_problem_heatmaps(
    csv_path: str,
    dataset_name: str,
    model_name: str,
    output_dir: str | None = None,
    problem_column: str = "problem",
    heatmap_column: str = "quantity_heatmap_black",
    shape_1_column: str = "heatmap_shape_1",
    shape_2_column: str = "heatmap_shape_2",
):
    """
    Generate one centered mean heatmap image for each problem.

    Each row heatmap can have a different shape. Heatmaps are center-aligned
    on a common canvas before averaging.
    """
    df = pd.read_csv(csv_path)

    required = [
        problem_column,
        heatmap_column,
        shape_1_column,
        shape_2_column,
    ]
    missing_columns = [col for col in required if col not in df.columns]
    if missing_columns:
        raise ValueError("Missing required columns: " +
                         ", ".join(missing_columns))

    if output_dir is None:
        output_dir = str(Path(csv_path).parent / "problem_heatmaps")

    base_output = Path(
        output_dir) / _sanitize_filename(dataset_name) / _sanitize_filename(model_name)
    base_output.mkdir(parents=True, exist_ok=True)

    # Build a global canvas so all problems use the same centered reference.
    global_h = 160  # int(df[shape_1_column].max())
    global_w = 160  # int(df[shape_2_column].max())

    saved_paths = []

    for problem, group in df.groupby(problem_column):
        problem_name = str(problem)
        sum_map = np.zeros((global_h, global_w), dtype=np.float32)
        count_map = np.zeros((global_h, global_w), dtype=np.float32)
        valid_rows = 0

        for _, row in group.iterrows():
            cell_value = row[heatmap_column]
            if pd.isna(cell_value):
                continue

            shape_1 = int(row[shape_1_column])
            shape_2 = int(row[shape_2_column])

            try:
                heatmap = _decode_heatmap(str(cell_value), shape_1, shape_2)
                centered_heatmap, valid_mask = _center_place(
                    heatmap, global_h, global_w)
            except Exception:
                continue

            sum_map += centered_heatmap
            count_map += valid_mask
            valid_rows += 1

        if valid_rows == 0:
            continue

        # Use discrete color levels to emphasize differences instead of smooth gradients.
        n_levels = 8
        valid_values = sum_map[count_map > 0]
        if valid_values.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin = float(np.min(valid_values))
            vmax = float(np.max(valid_values))
            if vmax <= vmin:
                vmax = vmin + 1.0

        boundaries = np.linspace(vmin, vmax, n_levels + 1)
        cmap = plt.get_cmap("turbo", n_levels)
        norm = BoundaryNorm(boundaries, cmap.N, clip=True)

        plt.figure(figsize=(7, 7))
        ax = plt.gca()
        image = plt.imshow(sum_map, cmap=cmap, norm=norm)
        ax.add_patch(
            Rectangle(
                (-0.5, -0.5),
                global_w,
                global_h,
                fill=False,
                edgecolor="cyan",
                linewidth=2.2,
                zorder=10,
                clip_on=False,
            )
        )
        ax.set_xlim(-0.5, global_w - 0.5)
        ax.set_ylim(global_h - 0.5, -0.5)
        plt.colorbar(image, boundaries=boundaries,
                     ticks=boundaries, fraction=0.046, pad=0.04)
        plt.title(
            f"Summed Mask Heatmap - {problem_name}\n{dataset_name} / {model_name} (n={valid_rows})"
        )
        plt.axis("off")
        plt.tight_layout()

        plot_name = f"{_sanitize_filename(problem_name)}_mean_mask_heatmap.png"
        plot_path = base_output / plot_name
        plt.savefig(plot_path, dpi=220, bbox_inches="tight")
        plt.close()

        saved_paths.append(str(plot_path))

    return saved_paths


def main():
    csv_path = "/Users/rheidner/framework-xai-optimization/results_imagewoof_insertion.csv"
    dataset_name = "imagewoof"
    model_name = "resnet"

    saved = generate_problem_heatmaps(
        csv_path=csv_path,
        dataset_name=dataset_name,
        model_name=model_name,
    )

    if saved:
        print("Saved heatmaps:")
        for path in saved:
            print(path)
    else:
        print("No heatmaps were generated. Check if CSV has valid encoded masks.")


if __name__ == "__main__":
    main()
