import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

from scipy.stats import kruskal
import scikit_posthocs as sp

# Mapeamento de nomes dos problemas
problem_mapping = {
    'guided_backprop_score': 'G. Backprop.',
    'guided_gradcam_score': 'G. Grad-CAM',
    'integrated_gradients_score': 'I. Gradients',
    'ipha_flavio_marcelo_v': 'IPHA-GA',
    'layer_gradcam_score': 'Grad-CAM',
    'menor_probabilidade_menor_area_inpaint_v': 'IPHA-RI',
    'menor_probabilidade_menor_area_v': 'IPHA-R',
    'poligono_e_perimetro_com_inpaint_v': 'IPHA-PI',
    'poligono_e_perimetro_v': 'IPHA-P',
    'saliency_score': 'Saliency Map'
}


def print_raw_data_structure(raw_data):
    for metric, datasets in raw_data.items():
        print(f"Metric: {metric}", sep=' ')
        for dataset, models in datasets.items():
            print(f"  Dataset: {dataset}", sep=' ')
            for model, problems in models.items():
                print(f"    Model: {model}", sep=' ')
                for problem, values in problems.items():
                    print(f"      Problem: {problem}")


def nemenyi_letters_by_problem(values_by_problem):
    """
    values_by_problem: dict {problema: [valores...]}
    Retorna dict {problema: letra}
    """
    problems = list(values_by_problem.keys())
    if len(problems) < 2:
        return {p: "A" for p in problems}

    arrays = [np.asarray(values_by_problem[p], dtype=float) for p in problems]

    _, p_kw = kruskal(*arrays)
    if p_kw > 0.05:
        return {p: "A" for p in problems}

    p_matrix = pd.DataFrame(sp.posthoc_nemenyi(arrays))
    p_matrix.index = problems
    p_matrix.columns = problems

    means = {p: float(np.mean(values_by_problem[p])) for p in problems}
    order = sorted(problems, key=lambda p: means[p], reverse=False)

    adjacency = {p: set() for p in problems}
    for i, p1 in enumerate(order):
        for p2 in order[i + 1:]:
            if p_matrix.loc[p1, p2] > 0.05:
                adjacency[p1].add(p2)
                adjacency[p2].add(p1)
                print(
                    f"Problemas {p1} e {p2} não são significativamente diferentes (p={p_matrix.loc[p1, p2]:.4f})")
            else:
                print(
                    f"Problemas {p1} e {p2} são significativamente diferentes (p={p_matrix.loc[p1, p2]:.4f})")

    visited = set()
    groups = []
    for p in order:
        if p in visited:
            continue
        stack = [p]
        comp = set()
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            comp.add(node)
            for n in adjacency[node]:
                if n not in visited:
                    stack.append(n)
        groups.append(sorted(comp, key=lambda x: means[x], reverse=True))

    labels = {}
    for idx, group in enumerate(groups):
        letter = chr(65 + idx)
        for p in group:
            labels[p] = letter

    for p in problems:
        labels.setdefault(p, chr(65 + len(groups)))

    return labels


def extract_model_dataset(folder_name):
    """Extrai modelo e dataset do nome da pasta."""
    parts = folder_name.replace('results_', '').split('_')

    if len(parts) >= 2:
        dataset = parts[0]  # cars, imagenette, etc
        # squeezenet10, resnet18, etc
        model = '_'.join(parts[1:-1]) if len(parts) > 2 else parts[1]
        return dataset, model

    return None, None


def generate_complete_heatmap(csv_paths):
    """Cria uma grade 3x3 de heatmaps para Score, Área e Tempo por Dataset."""

    # Estrutura para armazenar dados: metrics_data[metrica][dataset][problema][modelo] = valor
    metrics = ['Score', 'Área', 'Tempo']
    data_struct = {m: {} for m in metrics}
    raw_data = {m: {} for m in metrics}

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        df = df.head(10000)
        folder_name = Path(csv_path).parent.name
        dataset, model = extract_model_dataset(folder_name)

        if dataset is None:
            continue

        df['fi_important'] = df['original_score'] - \
            df['quantity_score_cutting_black']

        # Agrupar e extrair a média das três métricas simultaneamente
        # NOTA: Ajuste os nomes das colunas de 'area' e 'time' se forem diferentes no seu CSV
        grouped = df.groupby('problem').agg({
            'fi_important': 'mean',
            'quantity_area_black': 'mean',  # <-- AJUSTE O NOME DA COLUNA DE ÁREA SE NECESSÁRIO
            'time': 'mean'   # <-- AJUSTE O NOME DA COLUNA DE TEMPO SE NECESSÁRIO
        }).reset_index()

        for m in metrics:
            if dataset not in data_struct[m]:
                data_struct[m][dataset] = {}
                raw_data[m][dataset] = {}

        for metric, col_name in {
            "Score": "fi_important",
            "Área": "quantity_area_black",
            "Tempo": "time",
        }.items():
            raw_data[metric].setdefault(dataset, {})
            raw_data[metric][dataset].setdefault(model, {})

            for problem_name, group in df.groupby("problem"):
                problem_label = problem_mapping.get(problem_name, problem_name)
                raw_data[metric][dataset][model].setdefault(problem_label, [])
                raw_data[metric][dataset][model][problem_label].extend(
                    group[col_name].tolist())

        for idx, row in grouped.iterrows():
            problem = row['problem']

            for m, col_name in zip(metrics, ['fi_important', 'quantity_area_black', 'time']):
                if problem not in data_struct[m][dataset]:
                    data_struct[m][dataset][problem] = {}
                data_struct[m][dataset][problem][model] = row[col_name]

    # Cria significance_map para armazenar as letras de significância por problema e modelo
    significance_map = {}
    for metric in metrics:
        significance_map[metric] = {}
        for dataset in raw_data[metric]:
            significance_map[metric].setdefault(dataset, {})
            for model in raw_data[metric][dataset]:
                significance_map[metric][dataset].setdefault(model, {})
                for problem in raw_data[metric][dataset][model]:
                    values_by_problem = {
                        p: raw_data[metric][dataset][model].get(p, [])
                        for p in raw_data[metric][dataset][model]
                    }
                    letters = nemenyi_letters_by_problem(values_by_problem)
                    significance_map[metric][dataset][model] = letters

    # Calcular média geral do SCORE para manter a ordenação consistente de linhas
    problem_means = {}
    for dataset in data_struct['Score']:
        for problem, models_scores in data_struct['Score'][dataset].items():
            if problem not in problem_means:
                problem_means[problem] = []
            problem_means[problem].extend(models_scores.values())

    problem_order = sorted(problem_means.keys(), key=lambda p: np.mean(
        problem_means[p]), reverse=False)
    problem_order_mapped = [problem_mapping.get(p, p) for p in problem_order]

    # --- CONFIGURAÇÃO DA FIGURA 3x3 ---
    # --- CONFIGURAÇÃO DA FIGURA COMPACTA ---
    dataset_order = ['cars', 'imagewoof', 'imagenette']
    model_short_names = {'squeezenet10': 'SqueezeNet',
                         'resnet18': 'ResNet-18', 'vgg16bn': 'VGG-16 BN'}
    model_long_names = {'SqueezeNet': 'squeezenet10',
                        'ResNet-18': 'resnet18', 'VGG-16 BN': 'vgg16bn'}

    # Reduzimos a altura (figsize de 14 para 10) e aproximamos as linhas (hspace=0.08)
    fig, axes = plt.subplots(3, 3, figsize=(12, 12), sharey=True,
                             gridspec_kw={'wspace': 0.05, 'hspace': 0.08, 'bottom': 0.05})

    # Configurações de cores por métrica
    metric_configs = {
        'Score': {'cmap': 'viridis_r'},
        'Área':  {'cmap': 'viridis_r'},
        'Tempo': {'cmap': 'viridis_r'}
    }

    # Loop pelas Linhas (Métricas) e Colunas (Datasets)
    for row_idx, metric in enumerate(metrics):
        sorted_datasets = [(ds, data_struct[metric][ds])
                           for ds in dataset_order if ds in data_struct[metric]]

        all_values = [v for ds_data in data_struct[metric].values()
                      for p_data in ds_data.values() for v in p_data.values()]
        v_min, v_max = (0, 1) if metric == 'Score' else (
            min(all_values), max(all_values))

        for col_idx, (dataset, problems_data) in enumerate(sorted_datasets):
            ax = axes[row_idx, col_idx]

            df_heatmap = pd.DataFrame(problems_data).T
            df_heatmap.index = [problem_mapping.get(
                p, p) for p in df_heatmap.index]
            df_heatmap = df_heatmap.reindex(sorted(df_heatmap.columns), axis=1)
            df_heatmap = df_heatmap.reindex(problem_order_mapped)
            df_heatmap.columns = [model_short_names.get(
                c, c) for c in df_heatmap.columns]

            # criar anotação com valor + letra
            annot_df = pd.DataFrame(
                index=df_heatmap.index, columns=df_heatmap.columns)
            for problem in df_heatmap.index:
                for model_long_name in df_heatmap.columns:
                    model = model_long_names.get(
                        model_long_name, model_long_name)
                    value = df_heatmap.loc[problem, model_long_name]
                    letter = significance_map[metric][dataset].get(
                        model, {}).get(problem, '')
                    if letter == 'A':
                        annot_df.loc[problem,
                                     model_long_name] = f"{value:.3f} $\mathbf{{{letter}}}$"
                    else:
                        annot_df.loc[problem,
                                     model_long_name] = f"{value:.3f} {letter}"

            # cbar=False remove completamente as barras de valores inferiores
            sns.heatmap(
                df_heatmap,
                annot=annot_df.to_numpy(),
                fmt='',
                cmap=metric_configs[metric]['cmap'],
                vmin=v_min, vmax=v_max,
                cbar=False,
                ax=ax,
                linewidths=0.5,
                linecolor='white',
                annot_kws={"size": 10},
            )

            # Título do Dataset apenas na primeira linha de gráficos
            if row_idx == 0:
                ax.set_title(dataset.upper(), fontsize=13,
                             fontweight='bold', pad=12)

            # Identificador da Métrica no lado esquerdo da linha
            if col_idx == 0:
                ax.set_ylabel(metric, fontsize=14,
                              fontweight='bold', labelpad=15)
                ax.tick_params(axis='y', rotation=0, labelsize=10)
            else:
                ax.set_ylabel('')
                ax.yaxis.set_visible(False)

            # Ajustar rótulos do eixo X (Modelos) apenas na última linha
            if row_idx == 2:
                ax.tick_params(axis='x', rotation=15, labelsize=10)
            else:
                ax.set_xlabel('')
                ax.xaxis.set_visible(False)

    # O tight_layout recalcula os espaços para eliminar rebarbas brancas
    plt.tight_layout()
    plt.savefig('heatmap_complete_matrix.png', dpi=300, bbox_inches='tight')
    print("Matriz compactada e sem barras salva com sucesso!")
    plt.close()


# Uso com múltiplos CSVs de diferentes modelos e datasets
csv_files = [
    '/Users/rheidner/Downloads/results/results_cars_squeezenet10_article/results.csv',
    '/Users/rheidner/Downloads/results/results_cars_resnet18_article/results.csv',
    '/Users/rheidner/Downloads/results/results_cars_vgg16bn_article/results.csv',
    '/Users/rheidner/Downloads/results/results_imagewoof_squeezenet10_article/results.csv',
    '/Users/rheidner/Downloads/results/results_imagewoof_resnet18_article/results.csv',
    '/Users/rheidner/Downloads/results/results_imagewoof_vgg16bn_article/results.csv',
    '/Users/rheidner/Downloads/results/results_imagenette_squeezenet10_article/results.csv',
    '/Users/rheidner/Downloads/results/results_imagenette_resnet18_article/results.csv',
    '/Users/rheidner/Downloads/results/results_imagenette_vgg16bn_article/results.csv',
]

generate_complete_heatmap(csv_files)
