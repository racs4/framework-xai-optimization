from mestradolib.model import *
from mestradolib.problems import *
from mestradolib.inpaint import *
from mestradolib.utils import *
from mestradolib.visual import *
from mestradolib.run import *
import numpy as np


# n01440764	Tench (peixe)
# n02102040	English Springer (cachorro)
# n02979186	Cassette Player
# n03000684	Chain Saw
# n03028079	Church
# n03394916	French Horn
# n03417042	Garbage Truck
# n03425413	Gas Pump
# n03445777	Golf Ball
# n03888257	Parachute


if __name__ == "__main__":
    reproducibility()

    data = get_imagenette_loader()

    learner = get_learner(data, ds_name="imagenette")

    idxs = [1354]  # , 0, 60, 99, 95, 63, 65]
    # idxs = np.random.randint(0, len(data.train_ds), size=6000)
    problems = [
        MenorProbalidadeMenorAreaVectorized(
            4,
            2,
            problem_best_image=create_rectangle_best_image,
            problem_heatmap=create_rectangle_heatmap,
            problem_probability_heatmap=create_rectangle_probability_heatmap,
            problem_name="menor_probabilidade_menor_area_v",
            xl=None,
            xu=None,
            mutation=None,
            crossover=None,
        ),
        PoligonoEPerimetroVectorized(
            10,
            2,
            problem_best_image=create_polygonal_best_image,
            problem_heatmap=create_polygonal_heatmap,
            problem_probability_heatmap=create_polygonal_probability_heatmap,
            problem_name="poligono_e_perimetro_v",
        ),
        MenorProbalidadeMenorAreaInpaintVectorized(
            4,
            2,
            problem_best_image=create_rectangle_best_image,
            problem_heatmap=create_rectangle_heatmap,
            problem_probability_heatmap=create_rectangle_probability_heatmap,
            problem_name="menor_probabilidade_menor_area_inpaint_v",
        ),
        PoligonoEPerimetroInpaintVectorized(
            10,
            2,
            problem_best_image=create_polygonal_best_image,
            problem_heatmap=create_polygonal_heatmap,
            problem_probability_heatmap=create_polygonal_probability_heatmap,
            problem_name="poligono_e_perimetro_com_inpaint_v",
        ),
    ]
    apply_methods = [
        apply_black_heatmap,
        apply_inpainting_heatmap,
        apply_cutting_heatmap,
    ]

    folder_name = "results/results_after_1"
    run_test(folder_name, idxs, problems, learner, data)
