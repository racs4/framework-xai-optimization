# %%
import pandas as pd

df = pd.read_csv("results/results_imagenette/results.csv")
df

# %%
import matplotlib.pyplot as plt
import seaborn as sns

y_vars = [
    "best_score_black",
    "best_score_inpaint",
    "best_score_cutting_black",
    "quantity_score_black",
    "quantity_score_inpaint",
    "quantity_score_cutting_black",
    "probability_score_black",
    "probability_score_inpaint",
    "probability_score_cutting_black",
]

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 24))
axes = axes.flatten()

for i, y in enumerate(y_vars):
    sns.boxplot(x="problem", y=y, data=df, ax=axes[i])
    sns.stripplot(
        x="problem",
        y=y,
        data=df,
        ax=axes[i],
        color="black",
        jitter=True,
        size=5,
        alpha=0.1,
    )
    axes[i].set_title(f"Boxplot of {y} by Problem")
    axes[i].tick_params(axis="x", rotation=90)

plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
import seaborn as sns

y_vars = [
    "best_score_black",
    "best_score_inpaint",
    "best_score_cutting_black",
    "quantity_score_black",
    "quantity_score_inpaint",
    "quantity_score_cutting_black",
    "probability_score_black",
    "probability_score_inpaint",
    "probability_score_cutting_black",
]

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 24))
axes = axes.flatten()

for i, y in enumerate(y_vars):
    diff_by_problem = (
        df.assign(diff=(df["original_score"] - df[y]).abs())
        .groupby("problem")["diff"]
        .mean()
        .reset_index(name="mean_abs_diff")
    )
    diff_by_problem.plot.bar(
        x="problem", y="mean_abs_diff", legend=False, ax=axes[i], color="tab:blue"
    )
    axes[i].set_ylabel(f"FII {y}")
    axes[i].tick_params(axis="x", rotation=90)

plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
import seaborn as sns

y_vars = [
    "best_score_black",
    "best_score_inpaint",
    "best_score_cutting_black",
    "quantity_score_black",
    "quantity_score_inpaint",
    "quantity_score_cutting_black",
    "probability_score_black",
    "probability_score_inpaint",
    "probability_score_cutting_black",
]

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 24))
axes = axes.flatten()

for i, y in enumerate(y_vars):
    diff_by_problem = (
        df.assign(diff=((df["original_score"] - df[y]) / df["original_score"]))
        .groupby("problem")["diff"]
        .mean()
        .reset_index(name="mean_abs_diff")
    )
    diff_by_problem.plot.bar(
        x="problem", y="mean_abs_diff", legend=False, ax=axes[i], color="tab:blue"
    )
    axes[i].set_ylabel(f"FII {y}")
    axes[i].tick_params(axis="x", rotation=90)

plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
import seaborn as sns

y_vars = [
    "best_score_black",
    "best_score_inpaint",
    "best_score_cutting_black",
    "quantity_score_black",
    "quantity_score_inpaint",
    "quantity_score_cutting_black",
    "probability_score_black",
    "probability_score_inpaint",
    "probability_score_cutting_black",
]

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 24))
axes = axes.flatten()

for i, y in enumerate(y_vars):
    diff_by_problem = (
        df.assign(diff=(df["original_score"]))
        .groupby("problem")["diff"]
        .mean()
        .reset_index(name="mean_abs_diff")
    )
    diff_by_problem.plot.bar(
        x="problem", y="mean_abs_diff", legend=False, ax=axes[i], color="tab:blue"
    )
    axes[i].set_ylabel(f"FII {y}")
    axes[i].tick_params(axis="x", rotation=90)

plt.tight_layout()
plt.show()

# %%
y_vars = [
    "best_score_black",
    "best_score_inpaint",
    "quantity_score_black",
    "quantity_score_inpaint",
    "probability_score_black",
    "probability_score_inpaint",
]

y_image_names = [
    "best_black",
    "best_inpaint",
    "masked_black_quantity",
    "masked_inpaint_quantity",
    "masked_black_probability",
    "masked_inpaint_probability",
]

fig, axes = plt.subplots(nrows=24, ncols=2, figsize=(18, 72))
axes = axes.flatten()

unique_problems = df["problem"].unique()
num_problems = len(unique_problems)
num_vars = len(y_vars)
for j in range(num_problems):
    problem = unique_problems[j]
    # print(f"\nProblema: {problem}")
    df_problem = df[df["problem"] == problem]
    for i in range(num_vars):
        y = y_vars[i]
        y_image_name = y_image_names[i]
        min_idx = df_problem[y].idxmin()
        min_val = df_problem.loc[min_idx, y]
        min_idx = df_problem.loc[min_idx, "idx"]
        # print(f"{y}: índice com menor valor = {min_idx}, valor = {min_val}")
        # show image in results_test_6/{min_idx}/{problem}/{y_image_name}.png
        original_img_path = f"results_test_6/{min_idx}/original.png"
        img_path = f"results_test_6/{min_idx}/{problem}/{y_image_name}.png"
        img = plt.imread(img_path)
        axes[2 * (i + j * num_vars) + 0].imshow(plt.imread(original_img_path))
        axes[2 * (i + j * num_vars) + 0].set_title(f"Original")
        axes[2 * (i + j * num_vars) + 0].axis("off")
        axes[2 * (i + j * num_vars) + 1].imshow(img)
        axes[2 * (i + j * num_vars) + 1].set_title(
            f"{y} - Problema: {problem}\nÍndice: {min_idx} - Valor: {min_val:.4f}"
        )
        axes[2 * (i + j * num_vars) + 1].axis("off")

plt.tight_layout()
plt.show()

# %%
y_vars = [
    "best_score_black",
    "best_score_inpaint",
    "quantity_score_black",
    "quantity_score_inpaint",
    "probability_score_black",
    "probability_score_inpaint",
]

y_image_names = [
    "best_black",
    "best_inpaint",
    "masked_black_quantity",
    "masked_inpaint_quantity",
    "masked_black_probability",
    "masked_inpaint_probability",
]

fig, axes = plt.subplots(nrows=24, ncols=2, figsize=(18, 72))
axes = axes.flatten()

unique_problems = df["problem"].unique()
num_problems = len(unique_problems)
num_vars = len(y_vars)
for j in range(num_problems):
    problem = unique_problems[j]
    # print(f"\nProblema: {problem}")
    df_problem = df[df["problem"] == problem]
    for i in range(num_vars):
        y = y_vars[i]
        y_image_name = y_image_names[i]
        max_idx = df_problem[y].idxmax()
        max_val = df_problem.loc[max_idx, y]
        max_idx = df_problem.loc[max_idx, "idx"]
        # print(f"{y}: índice com menor valor = {max_idx}, valor = {max_val}")
        # show image in results_test_6/{max_idx}/{problem}/{y_image_name}.png
        original_img_path = f"results_test_6/{max_idx}/original.png"
        img_path = f"results_test_6/{max_idx}/{problem}/{y_image_name}.png"
        img = plt.imread(img_path)
        axes[2 * (i + j * num_vars) + 0].imshow(plt.imread(original_img_path))
        axes[2 * (i + j * num_vars) + 0].set_title(f"Original")
        axes[2 * (i + j * num_vars) + 0].axis("off")
        axes[2 * (i + j * num_vars) + 1].imshow(img)
        axes[2 * (i + j * num_vars) + 1].set_title(
            f"{y} - Problema: {problem}\nÍndice: {max_idx} - Valor: {max_val:.4f}"
        )
        axes[2 * (i + j * num_vars) + 1].axis("off")

plt.tight_layout()
plt.show()

# %%
y_vars = [
    "best_score_cutting_black",
    "quantity_score_cutting_black",
    "probability_score_cutting_black",
]

y_image_names = [
    "best_cutted_black",
    "masked_cutting_black_quantity",
    "masked_cutting_black_probability",
]

fig, axes = plt.subplots(nrows=12, ncols=2, figsize=(18, 36))
axes = axes.flatten()

unique_problems = df["problem"].unique()
num_problems = len(unique_problems)
num_vars = len(y_vars)
for j in range(num_problems):
    problem = unique_problems[j]
    # print(f"\nProblema: {problem}")
    df_problem = df[df["problem"] == problem]
    for i in range(num_vars):
        y = y_vars[i]
        y_image_name = y_image_names[i]
        min_idx = df_problem[y].idxmax()
        min_val = df_problem.loc[min_idx, y]
        min_idx = df_problem.loc[min_idx, "idx"]
        # print(f"{y}: índice com menor valor = {min_idx}, valor = {min_val}")
        # show image in results_test_6/{min_idx}/{problem}/{y_image_name}.png
        original_img_path = f"results_test_6/{min_idx}/original.png"
        img_path = f"results_test_6/{min_idx}/{problem}/{y_image_name}.png"
        img = plt.imread(img_path)
        axes[2 * (i + j * num_vars) + 0].imshow(plt.imread(original_img_path))
        axes[2 * (i + j * num_vars) + 0].set_title(f"Original")
        axes[2 * (i + j * num_vars) + 0].axis("off")
        axes[2 * (i + j * num_vars) + 1].imshow(img)
        axes[2 * (i + j * num_vars) + 1].set_title(
            f"{y} - Problema: {problem}\nÍndice: {min_idx} - Valor: {min_val:.4f}"
        )
        axes[2 * (i + j * num_vars) + 1].axis("off")

plt.tight_layout()
plt.show()

# %%
y_vars = [
    "best_score_cutting_black",
    "quantity_score_cutting_black",
    "probability_score_cutting_black",
]

y_image_names = [
    "best_cutted_black",
    "masked_cutting_black_quantity",
    "masked_cutting_black_probability",
]

fig, axes = plt.subplots(nrows=12, ncols=2, figsize=(18, 36))
axes = axes.flatten()

unique_problems = df["problem"].unique()
num_problems = len(unique_problems)
num_vars = len(y_vars)
for j in range(num_problems):
    problem = unique_problems[j]
    # print(f"\nProblema: {problem}")
    df_problem = df[df["problem"] == problem]
    for i in range(num_vars):
        y = y_vars[i]
        y_image_name = y_image_names[i]
        min_idx = df_problem[y].idxmin()
        min_val = df_problem.loc[min_idx, y]
        min_idx = df_problem.loc[min_idx, "idx"]
        # print(f"{y}: índice com menor valor = {min_idx}, valor = {min_val}")
        # show image in results_test_6/{min_idx}/{problem}/{y_image_name}.png
        original_img_path = f"results_test_6/{min_idx}/original.png"
        img_path = f"results_test_6/{min_idx}/{problem}/{y_image_name}.png"
        img = plt.imread(img_path)
        axes[2 * (i + j * num_vars) + 0].imshow(plt.imread(original_img_path))
        axes[2 * (i + j * num_vars) + 0].set_title(f"Original")
        axes[2 * (i + j * num_vars) + 0].axis("off")
        axes[2 * (i + j * num_vars) + 1].imshow(img)
        axes[2 * (i + j * num_vars) + 1].set_title(
            f"{y} - Problema: {problem}\nÍndice: {min_idx} - Valor: {min_val:.4f}"
        )
        axes[2 * (i + j * num_vars) + 1].axis("off")

plt.tight_layout()
plt.show()
