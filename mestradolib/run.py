import traceback

from pyrecorder.recorder import Recorder
from pyrecorder.writers.video import Video
from pymoo.visualization.scatter import Scatter
from joblib import Parallel, delayed
from tqdm import tqdm
from mestradolib.model import *
from mestradolib.problems import *
from mestradolib.inpaint import *
from mestradolib.utils import *
from mestradolib.visual import *
from mestradolib.literature import *
from wakepy import keep
import time


def get_score_and_save(
    img,
    res,
    threshold,
    create_function,
    apply_function,
    model_wrapper,
    original_score,
    original_idx,
    img_path,
    save_files=True,
    calculate_insertion_and_deletion_metrics=False,
):
    img, score, area, insertion_deletion_metrics = get_masked_image_with_score(
        img,
        res,
        threshold,
        create_function,
        apply_function,
        model_wrapper,
        original_score,
        original_idx,
        font_size=9,
        font_pos=(1, 2),
        calculate_insertion_and_deletion_metrics=calculate_insertion_and_deletion_metrics,
    )
    if save_files:
        img.save(img_path)
    return score, area, insertion_deletion_metrics


def process_problem(
    j,
    img,
    model_wrapper,
    original_score,
    original_idx,
    i,
    folder_name,
    problems,
    threshold=0.1,
    save_files=True,
):
    problem = problems[j]
    p_best_image = problem.problem_best_image
    p_heatmap = problem.problem_heatmap
    p_probability_heatmap = problem.problem_probability_heatmap
    p_name = problem.problem_name

    problem.add_img(img)
    if save_files:
        create_folder_if_not_exists(f"{folder_name}/{i}/{p_name}")

    # minimize rectangle problem with problem
    start = time.time()
    res = minimize_rectangle_problem(
        img, problem, model_wrapper, original_score, original_idx, 8
    )
    end = time.time()
    problem_time = end - start
    print(f"Tempo para {p_name} na imagem {i}: {end - start:.2f} segundos")

    save_video(
        folder_name,
        i,
        p_name,
        res,
        p_heatmap,
        img,
    )

    # save heatmap result on {folder_name}/{idx}/{problem_name}/heatmap_quantity.png
    if save_files:
        save_heatmap_result(
            img, res, p_heatmap, f"{folder_name}/{i}/{p_name}/heatmap_quantity.png"
        )

    best_score_black, best_area_black, best_insertion_deletion_metrics_black = get_score_and_save(
        img,
        res,
        0.01,
        p_best_image,
        apply_black_heatmap,
        model_wrapper,
        original_score,
        original_idx,
        f"{folder_name}/{i}/{p_name}/best_black.png",
        save_files,
    )

    best_score_cutting_black, best_area_cutting_black, best_insertion_deletion_metrics_cutting_black = get_score_and_save(
        img,
        res,
        0.01,
        p_best_image,
        apply_cutting_heatmap,
        model_wrapper,
        original_score,
        original_idx,
        f"{folder_name}/{i}/{p_name}/best_cutted_black.png",
        save_files,
    )

    best_score_inpaint, best_area_inpaint, best_insertion_deletion_metrics_inpaint = get_score_and_save(
        img,
        res,
        0.01,
        p_best_image,
        apply_inpainting_heatmap,
        model_wrapper,
        original_score,
        original_idx,
        f"{folder_name}/{i}/{p_name}/best_inpaint.png",
        save_files,
    )

    # save masked image on {folder_name}/{idx}/{problem_name}/heatmap_quantity_mask_black.png
    quantity_score_black, quantity_area_black, quantity_insertion_deletion_metrics_black = get_score_and_save(
        img,
        res,
        threshold,
        p_heatmap,
        apply_black_heatmap,
        model_wrapper,
        original_score,
        original_idx,
        f"{folder_name}/{i}/{p_name}/masked_black_quantity.png",
        save_files,
    )

    # save masked image on {folder_name}/{idx}/{problem_name}/heatmap_quantity_mask_black.png
    quantity_score_cutting_black, quantity_area_cutting_black, quantity_insertion_deletion_metrics_cutting_black = get_score_and_save(
        img,
        res,
        threshold,
        p_heatmap,
        apply_cutting_heatmap,
        model_wrapper,
        original_score,
        original_idx,
        f"{folder_name}/{i}/{p_name}/masked_cutting_black_quantity.png",
        save_files,
    )

    # save masked image on {folder_name}/{idx}/{problem_name}/heatmap_quantity_mask_inpaint.png
    quantity_score_inpaint, quantity_area_inpaint, quantity_insertion_deletion_metrics_inpaint = get_score_and_save(
        img,
        res,
        threshold,
        p_heatmap,
        apply_inpainting_heatmap,
        model_wrapper,
        original_score,
        original_idx,
        f"{folder_name}/{i}/{p_name}/masked_inpaint_quantity.png",
        save_files,
    )

    # save pareto front on {folder_name}/{idx}/{problem_name}/pareto_front.png
    if problem.n_obj > 1 and save_files:
        save_pareto_front(res, f"{folder_name}/{i}/{p_name}/pareto_front.png")

    # save probability heatmap on {folder_name}/{idx}/{problem_name}/heatmap_probability.png
    if save_files:
        save_heatmap_result(
            img,
            res,
            p_probability_heatmap,
            f"{folder_name}/{i}/{p_name}/heatmap_probability.png",
        )

    # save masked image on {folder_name}/{idx}/{problem_name}/heatmap_probability_mask_black.png
    probability_score_black, probability_area_black, probability_insertion_deletion_metrics_black = get_score_and_save(
        img,
        res,
        threshold,
        p_probability_heatmap,
        apply_black_heatmap,
        model_wrapper,
        original_score,
        original_idx,
        f"{folder_name}/{i}/{p_name}/masked_black_probability.png",
        save_files,
    )

    # save masked image on {folder_name}/{idx}/{problem_name}/heatmap_probability_mask_black.png
    probability_score_cutting_black, probability_area_cutting_black, probability_insertion_deletion_metrics_cutting_black = (
        get_score_and_save(
            img,
            res,
            threshold,
            p_probability_heatmap,
            apply_cutting_heatmap,
            model_wrapper,
            original_score,
            original_idx,
            f"{folder_name}/{i}/{p_name}/masked_cutting_black_probability.png",
            save_files,
        )
    )

    # save masked image on {folder_name}/{idx}/{problem_name}/heatmap_probability_mask_inpaint.png
    probability_score_inpaint, probability_area_inpaint, probability_insertion_deletion_metrics_inpaint = get_score_and_save(
        img,
        res,
        threshold,
        p_probability_heatmap,
        apply_inpainting_heatmap,
        model_wrapper,
        original_score,
        original_idx,
        f"{folder_name}/{i}/{p_name}/masked_inpaint_probability.png",
        save_files,
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
        problem_time,
        best_area_black,
        best_area_inpaint,
        best_area_cutting_black,
        quantity_area_black,
        quantity_area_inpaint,
        quantity_area_cutting_black,
        probability_area_black,
        probability_area_inpaint,
        probability_area_cutting_black,
    )


def save_video(folder_name, i, p_name, res, p_heatmap, img):
    with Recorder(Video(f"{folder_name}/{i}/{p_name}/optimization_video.mp4")) as rec:
        geracao = 0
        for entry in res.history:

            entry_res = Result()
            entry_res.X = entry.pop.get("X")

            masked_image = p_heatmap(img, entry_res)

            plt.figure(figsize=(8, 6))
            plt.imshow(np.array(img), cmap="gray", alpha=0.7)
            plt.imshow(masked_image, cmap="hot_r", alpha=0.5)
            plt.axis("off")
            plt.tight_layout()
            # save image
            plt.savefig(
                f"{folder_name}/{i}/{p_name}/optimization_frame_{geracao}.png",
                bbox_inches="tight",
            )
            rec.record()
            geracao += 1
        plt.close()


def run_test(
    folder_name,
    idxs,
    problems,
    model_wrapper,
    data,
    threshold=0.1,
    parallel=False,
    append=False,
    save_files=True,
):
    with keep.presenting():
        mode = "a" if append else "w"
        with open(f"{folder_name}/results.csv", mode) as f:
            if not append:
                f.write(
                    "idx,problem,original_score,best_score_black,best_score_inpaint,best_score_cutting_black,quantity_score_black,quantity_score_inpaint,quantity_score_cutting_black,probability_score_black,probability_score_inpaint,probability_score_cutting_black,threshold,time,best_area_black,best_area_inpaint,best_area_cutting_black,quantity_area_black,quantity_area_inpaint,quantity_area_cutting_black,probability_area_black,probability_area_inpaint,probability_area_cutting_black\n"
                )

            for i in tqdm(idxs):
                try:
                    # save original image with index on {folder_name}/{idx}/original.png
                    img = get_img(i, data, use_train=False)

                    npixels = img.size[0] * img.size[1]
                    for p in problems:
                        if p.dynamic_variables:
                            p.n_var = img.size[0] * img.size[1] // 16 + 1

                    if save_files:
                        create_folder_if_not_exists(f"{folder_name}/{i}")
                        img.save(f"{folder_name}/{i}/original.png")

                    # _, pred_idx_img, outputs_img = learner.predict(img)  # type: ignore
                    # prob_img_original = outputs_img[pred_idx_img].item()
                    # pred_idx_original = pred_idx_img

                    # original_score, original_idx = prob_img_original, pred_idx_original

                    original_score, original_idx = model_wrapper.get_img_score(
                        img)

                    results = []

                    if not parallel:
                        for j in range(len(problems)):
                            r = process_problem(
                                j,
                                img,
                                model_wrapper,
                                original_score,
                                original_idx,
                                i,
                                folder_name,
                                problems,
                                threshold,
                                save_files,
                            )
                            results.append(r)
                    else:
                        results = Parallel(n_jobs=-1)(
                            delayed(process_problem)(
                                j,
                                img,
                                model_wrapper,
                                original_score,
                                original_idx,
                                i,
                                folder_name,
                                problems,
                                threshold,
                            )
                            for j in range(len(problems))
                        )

                    area_sum = 0

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
                            problem_time,
                            best_area_black,
                            best_area_inpaint,
                            best_area_cutting_black,
                            quantity_area_black,
                            quantity_area_inpaint,
                            quantity_area_cutting_black,
                            probability_area_black,
                            probability_area_inpaint,
                            probability_area_cutting_black,
                        ) = r
                        # idx,problem,original_score,best_score_black,best_score_inpaint,best_score_cutting_black,quantity_score_black,quantity_score_inpaint,quantity_score_cutting_black,probability_score_black,probability_score_inpaint,probability_score_cutting_black,threshold,time,best_area_black,best_area_inpaint,best_area_cutting_black,quantity_area_black,quantity_area_inpaint,quantity_area_cutting_black,probability_area_black,probability_area_inpaint,probability_area_cutting_black\n"
                        f.write(
                            f"{i},{p_name},{original_score:.4f},{best_score_black:.4f},{best_score_inpaint:.4f},{best_score_cutting_black:.4f},{quantity_score_black:.4f},{quantity_score_inpaint:.4f},{quantity_score_cutting_black:.4f},{probability_score_black:.4f},{probability_score_inpaint:.4f},{probability_score_cutting_black:.4f},{threshold},{problem_time:.4f},{best_area_black:.4f},{best_area_inpaint:.4f},{best_area_cutting_black:.4f},{quantity_area_black:.4f},{quantity_area_inpaint:.4f},{quantity_area_cutting_black:.4f},{probability_area_black:.4f},{probability_area_inpaint:.4f},{probability_area_cutting_black:.4f}\n"
                        )
                        f.flush()
                        area_sum += quantity_area_black

                    if model_wrapper.run_literature:
                        print(
                            f"Calculating literature methods for image {i}...")
                        area_mean = 0
                        if len(results) > 0:
                            area_mean = area_sum / len(results)

                        if save_files:
                            create_folder_if_not_exists(
                                f"{folder_name}/{i}/literature")

                        literature_results = run_literature(
                            folder_name,
                            i,
                            "literature",
                            img,
                            model_wrapper,
                            original_score,
                            original_idx,
                            area_mean,
                            save_files,
                        )

                        for method in literature_results:
                            f.write(
                                f"{i},{method},{original_score:.4f},{literature_results[method][0]:.4f},,{literature_results[method][3]:.4f},{literature_results[method][0]:.4f},,{literature_results[method][3]:.4f},{literature_results[method][0]:.4f},,{literature_results[method][3]:.4f},0.05,{literature_results[method][1]:.4f},{literature_results[method][2]:.4f},,{literature_results[method][4]:.4f},{literature_results[method][2]:.4f},,{literature_results[method][4]:.4f},{literature_results[method][2]:.4f},,{literature_results[method][4]:.4f}\n"
                            )
                        f.flush()

                    plt.close("all")
                except Exception as e:
                    # mostrar stack

                    traceback.print_exc()

                    print(f"Error processing image {i}: {e}")
                    with open(f"{folder_name}/error.txt", "a") as error_file:
                        error_file.write(f"Error processing image {i}: {e}\n")
                    continue
