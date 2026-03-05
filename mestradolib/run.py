from joblib import Parallel, delayed
from tqdm import tqdm
from mestradolib.model import *
from mestradolib.problems import *
from mestradolib.inpaint import *
from mestradolib.utils import *
from mestradolib.visual import *
import numpy as np
import ctypes
from wakepy import keep
import time


def get_score_and_save(
    img,
    res,
    threshold,
    create_function,
    apply_function,
    learner,
    original_score,
    original_idx,
    img_path,
):
    img, score = get_masked_image_with_score(
        img,
        res,
        threshold,
        create_function,
        apply_function,
        learner,
        original_score,
        original_idx,
        font_size=9,
        font_pos=(1, 2),
    )
    img.save(img_path)
    return score


def process_problem(
    j, img, learner, original_score, original_idx, i, folder_name, problems
):
    problem = problems[j]
    p_best_image = problem.problem_best_image
    p_heatmap = problem.problem_heatmap
    p_probability_heatmap = problem.problem_probability_heatmap
    p_name = problem.problem_name

    problem.add_img(img)

    create_folder_if_not_exists(f"{folder_name}/{i}/{p_name}")

    # minimize rectangle problem with problem
    start = time.time()

    res = minimize_rectangle_problem(
        img, problem, learner, original_score, original_idx, 8
    )
    end = time.time()
    print(f"Tempo para {p_name} na imagem {i}: {end - start:.2f} segundos")

    # save heatmap result on {folder_name}/{idx}/{problem_name}/heatmap_quantity.png
    save_heatmap_result(
        img, res, p_heatmap, f"{folder_name}/{i}/{p_name}/heatmap_quantity.png"
    )

    best_score_black = get_score_and_save(
        img,
        res,
        0.01,
        p_best_image,
        apply_black_heatmap,
        learner,
        original_score,
        original_idx,
        f"{folder_name}/{i}/{p_name}/best_black.png",
    )

    best_score_cutting_black = get_score_and_save(
        img,
        res,
        0.01,
        p_best_image,
        apply_cutting_heatmap,
        learner,
        original_score,
        original_idx,
        f"{folder_name}/{i}/{p_name}/best_cutted_black.png",
    )

    best_score_inpaint = get_score_and_save(
        img,
        res,
        0.01,
        p_best_image,
        apply_inpainting_heatmap,
        learner,
        original_score,
        original_idx,
        f"{folder_name}/{i}/{p_name}/best_inpaint.png",
    )

    # save masked image on {folder_name}/{idx}/{problem_name}/heatmap_quantity_mask_black.png
    quantity_score_black = get_score_and_save(
        img,
        res,
        0.1,
        p_heatmap,
        apply_black_heatmap,
        learner,
        original_score,
        original_idx,
        f"{folder_name}/{i}/{p_name}/masked_black_quantity.png",
    )

    # save masked image on {folder_name}/{idx}/{problem_name}/heatmap_quantity_mask_black.png
    quantity_score_cutting_black = get_score_and_save(
        img,
        res,
        0.1,
        p_heatmap,
        apply_cutting_heatmap,
        learner,
        original_score,
        original_idx,
        f"{folder_name}/{i}/{p_name}/masked_cutting_black_quantity.png",
    )

    # save masked image on {folder_name}/{idx}/{problem_name}/heatmap_quantity_mask_inpaint.png
    quantity_score_inpaint = get_score_and_save(
        img,
        res,
        0.1,
        p_heatmap,
        apply_inpainting_heatmap,
        learner,
        original_score,
        original_idx,
        f"{folder_name}/{i}/{p_name}/masked_inpaint_quantity.png",
    )

    # save pareto front on {folder_name}/{idx}/{problem_name}/pareto_front.png
    save_pareto_front(res, f"{folder_name}/{i}/{p_name}/pareto_front.png")

    # save probability heatmap on {folder_name}/{idx}/{problem_name}/heatmap_probability.png
    save_heatmap_result(
        img,
        res,
        p_probability_heatmap,
        f"{folder_name}/{i}/{p_name}/heatmap_probability.png",
    )

    # save masked image on {folder_name}/{idx}/{problem_name}/heatmap_probability_mask_black.png
    probability_score_black = get_score_and_save(
        img,
        res,
        0.1,
        p_probability_heatmap,
        apply_black_heatmap,
        learner,
        original_score,
        original_idx,
        f"{folder_name}/{i}/{p_name}/masked_black_probability.png",
    )

    # save masked image on {folder_name}/{idx}/{problem_name}/heatmap_probability_mask_black.png
    probability_score_cutting_black = get_score_and_save(
        img,
        res,
        0.1,
        p_probability_heatmap,
        apply_cutting_heatmap,
        learner,
        original_score,
        original_idx,
        f"{folder_name}/{i}/{p_name}/masked_cutting_black_probability.png",
    )

    # save masked image on {folder_name}/{idx}/{problem_name}/heatmap_probability_mask_inpaint.png
    probability_score_inpaint = get_score_and_save(
        img,
        res,
        0.1,
        p_probability_heatmap,
        apply_inpainting_heatmap,
        learner,
        original_score,
        original_idx,
        f"{folder_name}/{i}/{p_name}/masked_inpaint_probability.png",
    )

    return (
        p_name,
        best_score_black,
        best_score_inpaint,
        best_score_cutting_black,
        quantity_score_black,
        quantity_score_inpaint,
        quantity_score_cutting_black,
        probability_score_black,
        probability_score_inpaint,
        probability_score_cutting_black,
    )


def  run_test(folder_name, idxs, problems, learner, data, parallel=False):
    with keep.presenting():
        with open(f"{folder_name}/results.csv", "w") as f:
            f.write(
                "idx,problem,original_score,best_score_black,best_score_inpaint,best_score_cutting_black,quantity_score_black,quantity_score_inpaint,quantity_score_cutting_black,probability_score_black,probability_score_inpaint,probability_score_cutting_black,threshold\n"
            )

            for i in tqdm(idxs):
                # save original image with index on {folder_name}/{idx}/original.png
                img = get_img(i, data, learner, use_train=False)

                create_folder_if_not_exists(f"{folder_name}/{i}")
                img.save(f"{folder_name}/{i}/original.png")

                _, pred_idx_img, outputs_img = learner.predict(img)  # type: ignore
                prob_img_original = outputs_img[pred_idx_img].item()
                pred_idx_original = pred_idx_img

                original_score, original_idx = prob_img_original, pred_idx_original

                original_score, original_idx = get_img_score(learner, img)

                # save results on an csv on {folder_name}/results.csv

                results = []

                if not parallel:
                    for j in range(len(problems)):
                        r = process_problem(
                            j,
                            img,
                            learner,
                            original_score,
                            original_idx,
                            i,
                            folder_name,
                            problems,
                        )
                        results.append(r)
                else:
                    results = Parallel(n_jobs=-1)(
                        delayed(process_problem)(
                            j, img, learner, original_score, original_idx, i, problems
                        )
                        for j in range(len(problems))
                    )

                for r in results:
                    (
                        p_name,
                        best_score_black,
                        best_score_inpaint,
                        best_score_cutting_black,
                        quantity_score_black,
                        quantity_score_inpaint,
                        quantity_score_cutting_black,
                        probability_score_black,
                        probability_score_inpaint,
                        probability_score_cutting_black,
                    ) = r
                    f.write(
                        f"{i},{p_name},{original_score:.4f},{best_score_black:.4f},{best_score_inpaint:.4f},{best_score_cutting_black:.4f},{quantity_score_black:.4f},{quantity_score_inpaint:.4f},{quantity_score_cutting_black:.4f},{probability_score_black:.4f},{probability_score_inpaint:.4f},{probability_score_cutting_black:.4f},0.1\n"
                    )
                    f.flush()

                plt.close("all")
