import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

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

def generate_latex_table_extended(csv_paths):
    all_results = []
    
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        
        # Agrupar por problema
        grouped = df.groupby('problem').agg({
            'quantity_area_cutting_black': ['mean', 'std'],
            'time': ['mean', 'std'],
            'quantity_score_cutting_black': ['mean', 'std']
        }).reset_index()
        
        for idx, row in grouped.iterrows():
            problem = row[('problem', '')]
            mean_area = row[('quantity_area_cutting_black', 'mean')]
            std_area = row[('quantity_area_cutting_black', 'std')]
            mean_time = row[('time', 'mean')]
            std_time = row[('time', 'std')]
            mean_score = row[('quantity_score_cutting_black', 'mean')]
            std_score = row[('quantity_score_cutting_black', 'std')]
            
            # Mapear nome do problema
            problem_name = problem_mapping.get(problem, problem)
            
            all_results.append({
                'Problem': problem_name,
                'Score': f'${mean_score:.4f} \\pm {std_score:.4f}$',
                'Area': f'${mean_area:.2f} \\pm {std_area:.2f}$',
                'Time (s)': f'${mean_time:.4f} \\pm {std_time:.4f}$',
                'score_csv': f'{mean_score:.3f} ± {std_score:.3f}',
                'area_csv': f'{mean_area:.2f} ± {std_area:.2f}',
                'time_csv': f'{mean_time:.3f} ± {std_time:.3f}',
                'score_value': mean_score,
                'area_value': mean_area,
                'time_value': mean_time,
            })
    
    results_df = pd.DataFrame(all_results)
    
    # Ordenar por score (maior para menor)
    results_df = results_df.sort_values('score_value', ascending=False).reset_index(drop=True)
    
    # Gerar LaTeX
    latex_table = results_df[['Problem', 'Area', 'Time (s)', 'Score']].to_latex(index=False, escape=False)
    
    # Adicionar título
    title = "Results for Cars Dataset with SqueezeNet1.0"
    latex_with_title = f"\\textbf{{{title}}}\n\n{latex_table}"

    with open('results_table.tex', 'w') as f:
        f.write(latex_with_title)
    
    # Salvar como CSV
    csv_output = results_df[['Problem', 'score_csv', 'area_csv', 'time_csv']].copy()
    csv_output.to_csv('results_table.csv', index=False, sep=';')
    print(f"CSV salvo em 'results_table.csv'")
    
    # Gerar imagem da tabela
    save_table_as_image(results_df, title, 'results_table.png')


def save_table_as_image(df, title, output_path):    
    # Encontrar índices dos valores extremos
    min_area_idx = df['area_value'].idxmin()
    min_time_idx = df['time_value'].idxmin()
    max_score_idx = df['score_value'].idxmax()
    
    # Preparar dados para a tabela
    table_data = []
    for idx, row in df.iterrows():
        table_data.append([
            row['Problem'],
            row['Score'],
            row['Area'],
            row['Time (s)']
        ])
    
    fig, ax = plt.subplots(figsize=(6,3))
    ax.axis('tight')
    ax.axis('off')
    
    # Criar tabela
    table = ax.table(
        cellText=table_data,
        colLabels=['Problem', 'Area', 'Time (s)', 'Score'],
        cellLoc='center',
        loc='center',
        colWidths=[0.4, 0.2, 0.2, 0.2]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1)
    
    # Remover todas as bordas
    for key, cell in table.get_celld().items():
        cell.set_linewidth(0)
        cell.set_edgecolor('none')
    
    # Adicionar apenas bordas horizontais (top e bottom)
    # Linha do header (top)
    for j in range(4):
        table[(0, j)].set_linewidth(1)
        table[(0, j)].set_edgecolor('black')
        table[(0, j)].set_linewidth(1.5)
        table[(0, j)].set_edgecolor('black')
        table[(0, j)].set_facecolor('white')
    
    # Linha separadora após header
    for j in range(4):
        table[(1, j)].set_linewidth(1)
        table[(1, j)].set_edgecolor('black')
    
    # Formatar células - sem cores, todas brancas
    for i in range(1, len(df) + 1):
        for j in range(4):
            table[(i, j)].set_facecolor('white')
            
            # Deixar em negrito os valores extremos
            if (i - 1 == min_area_idx and j == 3) or \
               (i - 1 == min_time_idx and j == 1) or \
               (i - 1 == max_score_idx and j == 2):
                table[(i, j)].set_text_props(weight='bold')
    
    # Adicionar título
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Imagem salva em '{output_path}'")
    plt.close()


csv_files = [
    'c:/Users/JARVIS/Downloads/results/results/results_cars_squeezenet10_article/results.csv',
]

generate_latex_table_extended(csv_files)