import base64
import re
import zlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def cell_to_array(cell, shape, dtype=np.float32):
    compressed = base64.b64decode(cell)
    raw = zlib.decompress(compressed)
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


def _sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_")


def _decode_curve(cell_value: str, n_steps: int) -> np.ndarray:
    curve = cell_to_array(cell_value, (n_steps,))
    return np.asarray(curve, dtype=np.float32).reshape(-1)


def plot_problem_curves(
    csv_path: str,
    dataset_name: str,
    model_name: str,
    plot_title: str,
    output_dir: str | None = None,
    curve_column: str = "quantity_insertion_curve_black",
    problem_column: str = "problem",
    n_steps_column: str = "n_steps",
):
    """
    Read one CSV, decode insertion curves, and generate one plot with all problems.

    X axis: step index
    Y axis: mean curve value on each step
    """
    df = pd.read_csv(csv_path)

    missing_columns = [
        col
        for col in [problem_column, curve_column, n_steps_column]
        if col not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing_columns)
        )

    if output_dir is None:
        output_dir = str(Path(csv_path).parent / "insertion_curve_plots")

    base_output = Path(
        output_dir) / _sanitize_filename(dataset_name) / _sanitize_filename(model_name)
    base_output.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    problem_curves: list[tuple[str, np.ndarray]] = []

    for problem, group in df.groupby(problem_column):
        problem_name = str(problem)
        curves = []

        for _, row in group.iterrows():
            cell_value = row[curve_column]
            if pd.isna(cell_value):
                continue

            n_steps = int(row[n_steps_column])

            try:
                curve = _decode_curve(str(cell_value), n_steps)
                curves.append(curve)
            except Exception:
                continue

        if not curves:
            continue

        min_len = min(len(curve) for curve in curves)
        aligned_curves = np.vstack([curve[:min_len] for curve in curves])
        mean_curve = np.mean(aligned_curves, axis=0)

        problem_curves.append((problem_name, mean_curve))

    if not problem_curves:
        return saved_paths

    global_min_len = min(len(curve) for _, curve in problem_curves)
    steps = np.arange(global_min_len)

    colors = plt.get_cmap("tab20")(np.linspace(0, 1, 20))
    line_styles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "+", "x"]
    mark_every = max(global_min_len // 12, 1)

    plt.figure(figsize=(12, 7))

    for idx, (problem_name, mean_curve) in enumerate(sorted(problem_curves, key=lambda x: x[0])):
        color = colors[idx % len(colors)]
        line_style = line_styles[idx % len(line_styles)]
        marker = markers[(idx // len(line_styles)) % len(markers)]
        plt.plot(
            steps,
            mean_curve[:global_min_len],
            label=problem_name,
            color=color,
            linestyle=line_style,
            marker=marker,
            markevery=mark_every,
            linewidth=2,
            markersize=4,
            alpha=0.95,
        )

    plt.title(
        f"Mean Insertion Curves by Problem\n{dataset_name} / {model_name}")
    plt.xlabel("Step")
    plt.ylabel("Mean value")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    plot_name = f"{plot_title}.png"
    plot_path = base_output / plot_name
    plt.savefig(plot_path, dpi=220)
    plt.close()

    saved_paths.append(str(plot_path))

    return saved_paths


def main():
    csv_path = "/Users/rheidner/framework-xai-optimization/results_imagewoof_insertion.csv"
    dataset_name = "imagewoof"
    model_name = "resnet"

    saved_insertion = plot_problem_curves(
        csv_path=csv_path,
        dataset_name=dataset_name,
        model_name=model_name,
        plot_title="all_problems_mean_insertion_curves_imagewoof",
    )

    saved_deletion = plot_problem_curves(
        csv_path=csv_path,
        dataset_name=dataset_name,
        model_name=model_name,
        curve_column="quantity_deletion_curve_black",
        plot_title="all_problems_mean_deletion_curves_imagewoof",
    )

    if saved_insertion:
        print("Saved insertion plots:")
        for path in saved_insertion:
            print(path)
    else:
        print(
            "No insertion plots were generated. Check if the CSV has valid encoded curves.")

    if saved_deletion:
        print("Saved deletion plots:")
        for path in saved_deletion:
            print(path)
    else:
        print(
            "No deletion plots were generated. Check if the CSV has valid encoded curves.")


if __name__ == "__main__":
    main()
