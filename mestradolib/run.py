import copy
import traceback

import scipy
from pyrecorder.recorder import Recorder
from pyrecorder.writers.video import Video
from pymoo.visualization.scatter import Scatter
from joblib import Parallel, delayed
from scipy import stats
from tqdm import tqdm
from mestradolib.grad_eclip import get_grad_eclip_poligons
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
):
    img, score, area = get_masked_image_with_score(
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
    )
    if save_files:
        img.save(img_path)
    return score, area


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
    with_grad_start=False
):
    problem = problems[j]
    p_best_image = problem.problem_best_image
    p_heatmap = problem.problem_heatmap
    p_probability_heatmap = problem.problem_probability_heatmap
    p_name = problem.problem_name

    poligons_extraction_start = 0
    poligons_extraction_end = 0
    poligons = []
    if with_grad_start:
        assert isinstance(model_wrapper, ModelWrapperOpenClip)

        poligons_extraction_start = time.time()
        poligons = get_grad_eclip_poligons(img, model_wrapper, threshold, problem.n_var)
        poligons_extraction_end = time.time()

        p_name += f" with EGrad"

        print(f"Pré-processamento para {p_name} na imagem {i}: {poligons_extraction_end - poligons_extraction_start:.2f} segundos")

    problem.add_img(img)
    if save_files:
        create_folder_if_not_exists(f"{folder_name}/{i}/{p_name}")

    # minimize rectangle problem with problem
    start = time.time()
    res = minimize_rectangle_problem(
        img, problem, model_wrapper, original_score, original_idx, 8, with_grad_start, poligons
    )
    end = time.time()
    problem_time = end - start
    poligons_extraction_time = poligons_extraction_end - poligons_extraction_start
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

    best_score_black, best_area_black = get_score_and_save(
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

    best_score_cutting_black, best_area_cutting_black = get_score_and_save(
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

    best_score_inpaint, best_area_inpaint = get_score_and_save(
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
    quantity_score_black, quantity_area_black = get_score_and_save(
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
    quantity_score_cutting_black, quantity_area_cutting_black = get_score_and_save(
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
    quantity_score_inpaint, quantity_area_inpaint = get_score_and_save(
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
    probability_score_black, probability_area_black = get_score_and_save(
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
    probability_score_cutting_black, probability_area_cutting_black = (
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
    probability_score_inpaint, probability_area_inpaint = get_score_and_save(
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
        poligons_extraction_time
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
    with_grad_start=False
):
    with keep.presenting():
        mode = "a" if append else "w"
        with open(f"{folder_name}/results.csv", mode) as f:
            if not append:
                f.write(
                    "idx,problem,original_score,best_score_black,best_score_inpaint,best_score_cutting_black,quantity_score_black,quantity_score_inpaint,quantity_score_cutting_black,probability_score_black,probability_score_inpaint,probability_score_cutting_black,threshold,time,best_area_black,best_area_inpaint,best_area_cutting_black,quantity_area_black,quantity_area_inpaint,quantity_area_cutting_black,probability_area_black,probability_area_inpaint,probability_area_cutting_black,egrad_poligons_time\n"
                )

            for i in tqdm(idxs):
                try:
                    # save original image with index on {folder_name}/{idx}/original.png
                    img = get_img(i, data, use_train=False)

                    npixels = img.size[0] * img.size[1]
                    for p in problems:
                        if p.dynamic_variables:
                            p.n_var = npixels // 16 + 1

                            if p.n_var % 2 != 0:
                                p.n_var += 1

                    if save_files:
                        create_folder_if_not_exists(f"{folder_name}/{i}")
                        img.save(f"{folder_name}/{i}/original.png")

                    # _, pred_idx_img, outputs_img = learner.predict(img)  # type: ignore
                    # prob_img_original = outputs_img[pred_idx_img].item()
                    # pred_idx_original = pred_idx_img

                    # original_score, original_idx = prob_img_original, pred_idx_original

                    original_score, original_idx = model_wrapper.get_img_score(img)

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
                                with_grad_start
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
                                with_grad_start
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
                            egrad_poligons_time
                        ) = r
                        # idx,problem,original_score,best_score_black,best_score_inpaint,best_score_cutting_black,quantity_score_black,quantity_score_inpaint,quantity_score_cutting_black,probability_score_black,probability_score_inpaint,probability_score_cutting_black,threshold,time,best_area_black,best_area_inpaint,best_area_cutting_black,quantity_area_black,quantity_area_inpaint,quantity_area_cutting_black,probability_area_black,probability_area_inpaint,probability_area_cutting_black,egrad_poligons_time\n"
                        f.write(
                            f"{i},{p_name},{original_score:.4f},{best_score_black:.4f},{best_score_inpaint:.4f},{best_score_cutting_black:.4f},{quantity_score_black:.4f},{quantity_score_inpaint:.4f},{quantity_score_cutting_black:.4f},{probability_score_black:.4f},{probability_score_inpaint:.4f},{probability_score_cutting_black:.4f},{threshold},{problem_time:.4f},{best_area_black:.4f},{best_area_inpaint:.4f},{best_area_cutting_black:.4f},{quantity_area_black:.4f},{quantity_area_inpaint:.4f},{quantity_area_cutting_black:.4f},{probability_area_black:.4f},{probability_area_inpaint:.4f},{probability_area_cutting_black:.4f},{egrad_poligons_time:.4f}\n"
                        )
                        f.flush()
                        area_sum += quantity_area_black

                    if model_wrapper.run_literature:
                        print(f"Calculating literature methods for image {i}...")
                        area_mean = 0
                        if len(results) > 0:
                            area_mean = area_sum / len(results)

                        if save_files:
                            create_folder_if_not_exists(f"{folder_name}/{i}/literature")

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

def print_estimated_n(print_text, results, field):
    print("-"*100)
    print(print_text, calculate_estimated_n(results, field))

def summarize_results(folder_name):
    results = dict()
    process_result_file(folder_name, results)

    sumarize_by_idx(folder_name, results)
    sumarize_by_problem(folder_name, results)

    print_estimated_n("Estimade n for time: ", results, "time")
    print_estimated_n("Estimade n for time (total): ", results, "total_time")
    print_estimated_n("Estimade n for FII: ", results, "fii")

def t_paired_test(diferences_by_idx):
    sd_diference = np.std(diferences_by_idx)
    z_alpha = 1.96
    z_beta = 1.282
    min_target_time_diference = 0.1

    # Estimated number of images needed to execute a main experiment
    estimated_n = ((z_alpha + z_beta) ** 2) * sd_diference ** 2 / min_target_time_diference ** 2

    return float(estimated_n)

def normal_approximation_central_limit_theorem(diferences_by_idx):
    sigma = np.std(diferences_by_idx)  # Estimated standard deviation of the population
    e = 0.1  # Tolerable margin error
    confidence = 0.95

    # Critical value Z for 95% confidence
    z = stats.norm.ppf(1 - (1 - confidence) / 2)

    n = (z * sigma / e) ** 2

    return float(n)

def calculate_estimated_n(results, field):
    diferences_by_idx = []
    for key, problems in results.items():
        diferences = [
            float(np.mean(problems["IPHA-R"][field])) - float(np.mean(problems["IPHA-R with EGrad"][field])),
            float(np.mean(problems["IPHA-P"][field])) - float(np.mean(problems["IPHA-P with EGrad"][field])),
            float(np.mean(problems["IPHA-RI"][field])) - float(np.mean(problems["IPHA-RI with EGrad"][field])),
            float(np.mean(problems["IPHA-PI"][field])) - float(np.mean(problems["IPHA-PI with EGrad"][field])),
            float(np.mean(problems["IPHA-GA"][field])) - float(np.mean(problems["IPHA-GA with EGrad"][field]))
        ]

        diferences_by_idx.append(sum(diferences) / len(diferences))

    stat, p_value = stats.shapiro(diferences_by_idx)
    print(f"Statistic ({field}): {stat:.5f}, p-value ({field}): {p_value:.5f}")

    if p_value > 0.05:
        return t_paired_test(diferences_by_idx)
    else:
        return normal_approximation_central_limit_theorem(diferences_by_idx)

def sumarize_by_idx(folder_name, results):
    with open(f"{folder_name}/results_summary_by_idx.csv", "w") as summary_file:
        summary_file.write(
            "idx,"
            "problem,"
            "fii (mean),"
            "fii (Margin Error),"
            "fii (median),"
            "fii (standard deviation),"
            "fii (Q1),"
            "fii (Q3),"
            "original_score,"
            # "best_score_black,"
            # "best_score_inpaint,"
            "best_score_cutting_black,"
            # "quantity_score_black,"
            # "quantity_score_inpaint,"
            # "quantity_score_cutting_black,"
            # "probability_score_black,"
            # "probability_score_inpaint,"
            # "probability_score_cutting_black,"
            # "threshold,"
            "time (mean),"
            "time (Margin Error),"
            "time (median),"
            "time (standard deviation),"
            "time (Q1),"
            "time (Q3),"
            # "best_area_black,"
            # "best_area_inpaint,"
            "best_area_cutting_black,"
            # "quantity_area_black,"
            # "quantity_area_inpaint,"
            # "quantity_area_cutting_black,"
            # "probability_area_black,"
            # "probability_area_inpaint,"
            # "probability_area_cutting_black,"
            "egrad_poligons_time (mean),"
            "egrad_poligons_time (Margin Error),"
            "egrad_poligons_time (median),"
            "egrad_poligons_time (standard deviation),"
            "egrad_poligons_time (Q1),"
            "egrad_poligons_time (Q3),"
            "total_time (mean),"
            "total_time (Margin Error),"
            "total_time (median),"
            "total_time (standard deviation),"
            "total_time (Q1),"
            "total_time (Q3)\n"
        )


        for key, problems in results.items():
            for problem, fields in problems.items():
                summary_file.write(f"{key}")
                summary_file.write(f",{problem}")

                write_cells(fields, summary_file)

                summary_file.write(f"\n")

        summary_file.close()


def sumarize_by_problem(folder_name, results):
    with open(f"{folder_name}/results_summary_by_problem.csv", "w") as summary_file:
        summary_file.write(
            "problem,"
            "fii (mean),"
            "fii (Margin Error),"
            "fii (median),"
            "fii (standard deviation),"
            "fii (Q1),"
            "fii (Q3),"
            "original_score,"
            # "best_score_black,"
            # "best_score_inpaint,"
            "best_score_cutting_black,"
            # "quantity_score_black,"
            # "quantity_score_inpaint,"
            # "quantity_score_cutting_black,"
            # "probability_score_black,"
            # "probability_score_inpaint,"
            # "probability_score_cutting_black,"
            # "threshold,"
            "time (mean),"
            "time (Margin Error),"
            "time (median),"
            "time (standard deviation),"
            "time (Q1),"
            "time (Q3),"
            # "best_area_black,"
            # "best_area_inpaint,"
            "best_area_cutting_black,"
            # "quantity_area_black,"
            # "quantity_area_inpaint,"
            # "quantity_area_cutting_black,"
            # "probability_area_black,"
            # "probability_area_inpaint,"
            # "probability_area_cutting_black,"
            "egrad_poligons_time (mean),"
            "egrad_poligons_time (Margin Error),"
            "egrad_poligons_time (median),"
            "egrad_poligons_time (standard deviation),"
            "egrad_poligons_time (Q1),"
            "egrad_poligons_time (Q3),"
            "total_time (mean),"
            "total_time (Margin Error),"
            "total_time (median),"
            "total_time (standard deviation),"
            "total_time (Q1),"
            "total_time (Q3)\n"
        )

        problem_values = {}
        for key, problems in results.items():
            for problem, fields in problems.items():
                if problem not in problem_values.keys():
                    problem_values[problem] = {}

                for field, values in fields.items():
                    if field not in problem_values[problem].keys():
                        problem_values[problem][field] = copy.deepcopy(values)
                    else:
                        problem_values[problem][field] += copy.deepcopy(values)

        for problem, values in problem_values.items():
            summary_file.write(f"{problem}")

            write_cells(values, summary_file)

            summary_file.write(f"\n")

        summary_file.close()

def write_cells(fields, summary_file):
    for field, values in fields.items():
        if values:
            mean = sum(values) / len(values)
        else:
            mean = -1

        cell = f"{mean:.4f}" if mean >= 0 else " - "

        if field in ("fii", "egrad_poligons_time", "time", "total_time") and values:
            cl = 0.95 # Confidence Level

            q1, median, q3 = np.percentile(values, [25, 50, 75], axis=0, method="nearest")
            standard_deviation = np.std(values)

            t = scipy.stats.t.ppf((1 + cl) / 2, len(values))
            margin_error = t * (standard_deviation / np.sqrt(len(values))) # Margin Error to 95% confidence interval

            cell += f",{float(margin_error):.4f},{median:.4f},{float(standard_deviation):.4f},{q1:.4f},{q3:.4f}"

        summary_file.write(f",{cell}")

def process_result_file(folder_name, results):
    with open(f"{folder_name}/results.csv", "r") as results_file:
        for result in results_file:
            (
                idx,
                problem,
                original_score,
                best_score_black,
                best_score_inpaint,
                best_score_cutting_black,
                quantity_score_black,
                quantity_score_inpaint,
                quantity_score_cutting_black,
                probability_score_black,
                probability_score_inpaint,
                probability_score_cutting_black,
                threshold,
                time,
                best_area_black,
                best_area_inpaint,
                best_area_cutting_black,
                quantity_area_black,
                quantity_area_inpaint,
                quantity_area_cutting_black,
                probability_area_black,
                probability_area_inpaint,
                probability_area_cutting_black,
                egrad_poligons_time
            ) = result.split(",")

            if idx == "idx":
                continue

            key = f"{idx}"

            if key not in results.keys():
                results[key] = {}

            if problem not in results[key].keys():
                results[key][problem] = {}

            values = {
                "fii": abs(float(original_score) - float(best_score_cutting_black)),
                "original_score": original_score,
                # "best_score_black": best_score_black,
                # "best_score_inpaint": best_score_inpaint,
                "best_score_cutting_black": best_score_cutting_black,
                # "quantity_score_black": quantity_score_black,
                # "quantity_score_inpaint": quantity_score_inpaint,
                # "quantity_score_cutting_black": quantity_score_cutting_black,
                # "probability_score_black": probability_score_black,
                # "probability_score_inpaint": probability_score_inpaint,
                # "probability_score_cutting_black": probability_score_cutting_black,
                # "threshold": threshold,
                "time": time,
                # "best_area_black": best_area_black,
                # "best_area_inpaint": best_area_inpaint,
                "best_area_cutting_black": best_area_cutting_black,
                # "quantity_area_black": quantity_area_black,
                # "quantity_area_inpaint": quantity_area_inpaint,
                # "quantity_area_cutting_black": quantity_area_cutting_black,
                # "probability_area_black": probability_area_black,
                # "probability_area_inpaint": probability_area_inpaint,
                # "probability_area_cutting_black": probability_area_cutting_black,
                "egrad_poligons_time": egrad_poligons_time,
                "total_time": float(time) + float(egrad_poligons_time)
            }

            for field, value in values.items():
                if field not in results[key][problem].keys():
                    results[key][problem][field] = []

                results[key][problem][field].append(float(value))

        results_file.close()
