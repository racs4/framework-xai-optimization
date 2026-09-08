from mestradolib.model import *
from mestradolib.problems import *
from mestradolib.inpaint import *
from mestradolib.utils import *
from mestradolib.visual import *
from mestradolib.run import *
import numpy as np
from fastai.vision.all import vgg16_bn, resnet18, squeezenet1_0

if __name__ == "__main__":
    reproducibility()

    with keep.presenting():
        data = get_imagenette_loader()

        print("Data size:", len(data.valid_ds))

        # model_arch = {
        #     "vgg16_bn": vgg16_bn,
        #     "resnet18": resnet18,
        #     "squeezenet1_0": squeezenet1_0,
        # }[learner_name]

        learner = get_learner(
            data, model_arch=resnet18, name="resnet18", ds_name="imagenette"
        )
        # model_wrapper = ModelWrapperFastAI(learner)
        # model_wrapper = ModelWrapperClip("openai/clip-vit-base-patch32", labels=["horn", "head"])
        model_wrapper = ModelWrapperOpenClip(labels=["horn", "head"])

        idxs = [337]  # [1130, 0, 60, 99, 95, 63, 65, 455]
        # idxs = np.random.randint(0, len(data.valid_ds), size=200)
        # idxs = list(range(1000, len(data.valid_ds)))
        problems = [
            MenorProbalidadeMenorAreaVectorized(
                4,
                2,
                problem_best_image=create_rectangle_best_image,
                problem_heatmap=create_rectangle_heatmap,
                problem_probability_heatmap=create_rectangle_probability_heatmap,
                problem_name="menor_probabilidade_menor_area_v",
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
            IPHAFlavioMarceloVectorized(
                problem_best_image=create_mask_best_image,
                problem_heatmap=create_mask_heatmap,
                problem_probability_heatmap=create_mask_probability_heatmap,
                problem_name="ipha_flavio_marcelo_v",
            ),
        ]
        apply_methods = [
            apply_black_heatmap,
            apply_inpainting_heatmap,
            apply_cutting_heatmap,
        ]

        learner_folder = "clip"

        folder_name = f"results/results_imagenette_{learner_folder}"
        create_folder_if_not_exists(folder_name)

        run_test(
            folder_name,
            idxs,
            problems,
            model_wrapper,
            data,
            threshold=0.1,
            parallel=False,
            append=True,
            save_files=True,
        )
