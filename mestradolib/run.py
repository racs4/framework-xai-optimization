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
        save_files=save_files,
        calculate_insertion_and_deletion_metrics=calculate_insertion_and_deletion_metrics,
    )
    if save_files:
        img.save(img_path)
        del_imgs_to_save = insertion_deletion_metrics.get("del_imgs_to_save", [])
        ins_imgs_to_save = insertion_deletion_metrics.get("ins_imgs_to_save", [])
        del_path = img_path.rsplit("/", 1)
        if len(del_path) == 2:
            del_path = f"{del_path[0]}/del/{del_path[1]}"
        else:
            del_path = f"del/{img_path}"
        create_folder_if_not_exists(del_path.rsplit("/", 1)[0])
        ins_path = img_path.rsplit("/", 1)
        if len(ins_path) == 2:
            ins_path = f"{ins_path[0]}/ins/{ins_path[1]}"
        else:
            ins_path = f"ins/{img_path}"
        create_folder_if_not_exists(ins_path.rsplit("/", 1)[0])
        for idx, del_img in enumerate(del_imgs_to_save):
            del_img.save(del_path.replace(".png", f"_del_{idx}.png"))
        for idx, ins_img in enumerate(ins_imgs_to_save):
            ins_img.save(ins_path.replace(".png", f"_ins_{idx}.png"))
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
    calculate_insertion_and_deletion_metrics=False
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
        calculate_insertion_and_deletion_metrics
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
        calculate_insertion_and_deletion_metrics
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
        calculate_insertion_and_deletion_metrics
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
        calculate_insertion_and_deletion_metrics
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
        calculate_insertion_and_deletion_metrics
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
        calculate_insertion_and_deletion_metrics
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
        calculate_insertion_and_deletion_metrics
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
            calculate_insertion_and_deletion_metrics
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
        calculate_insertion_and_deletion_metrics
    )

    return {
        "p_name": p_name,
        "best_score_black": best_score_black,
        "best_score_inpaint": best_score_inpaint,
        "best_score_cutting_black": best_score_cutting_black,
        "quantity_score_black": quantity_score_black,
        "quantity_score_inpaint": quantity_score_inpaint,
        "quantity_score_cutting_black": quantity_score_cutting_black,
        "probability_score_black": probability_score_black,
        "probability_score_inpaint": probability_score_inpaint,
        "probability_score_cutting_black": probability_score_cutting_black,
        "problem_time": problem_time,
        "best_area_black": best_area_black,
        "best_area_inpaint": best_area_inpaint,
        "best_area_cutting_black": best_area_cutting_black,
        "quantity_area_black": quantity_area_black,
        "quantity_area_inpaint": quantity_area_inpaint,
        "quantity_area_cutting_black": quantity_area_cutting_black,
        "probability_area_black": probability_area_black,
        "probability_area_inpaint": probability_area_inpaint,
        "probability_area_cutting_black": probability_area_cutting_black,
        "quantity_insertion_deletion_metrics_black": quantity_insertion_deletion_metrics_black,
        "quantity_insertion_deletion_metrics_inpaint": quantity_insertion_deletion_metrics_inpaint,
        "quantity_insertion_deletion_metrics_cutting_black": quantity_insertion_deletion_metrics_cutting_black,
        "probability_insertion_deletion_metrics_black": probability_insertion_deletion_metrics_black,
        "probability_insertion_deletion_metrics_inpaint": probability_insertion_deletion_metrics_inpaint,
        "probability_insertion_deletion_metrics_cutting_black": probability_insertion_deletion_metrics_cutting_black,
    }


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
    calculate_insertion_and_deletion_metrics=False
):
    with keep.presenting():
        mode = "a" if append else "w"
        with open(f"{folder_name}/results.csv", mode) as f:
            if not append:
                f.write(generate_csv_header())

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
                                calculate_insertion_and_deletion_metrics
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
                        quantity_area_black = r['quantity_area_black']
                        # idx,problem,original_score,best_score_black,best_score_inpaint,best_score_cutting_black,quantity_score_black,quantity_score_inpaint,quantity_score_cutting_black,probability_score_black,probability_score_inpaint,probability_score_cutting_black,threshold,time,best_area_black,best_area_inpaint,best_area_cutting_black,quantity_area_black,quantity_area_inpaint,quantity_area_cutting_black,probability_area_black,probability_area_inpaint,probability_area_cutting_black\n"
                        f.write(generate_csv_line_methods(i, original_score, r))
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
                            calculate_insertion_and_deletion_metrics=calculate_insertion_and_deletion_metrics
                        )

                        for method in literature_results:
                            f.write(generate_csv_line_literature(i, method, original_score, literature_results))
                        f.flush()

                    plt.close("all")
                except Exception as e:
                    # mostrar stack

                    traceback.print_exc()

                    print(f"Error processing image {i}: {e}")
                    with open(f"{folder_name}/error.txt", "a") as error_file:
                        error_file.write(f"Error processing image {i}: {e}\n")
                    continue

def generate_csv_header():
    return (
        f"idx,problem,original_score,best_score_black,best_score_inpaint,"
        f"best_score_cutting_black,quantity_score_black,quantity_score_inpaint,"
        f"quantity_score_cutting_black,probability_score_black,probability_score_inpaint,"
        f"probability_score_cutting_black,threshold,time,"
        f"best_area_black,best_area_inpaint,best_area_cutting_black,"
        f"quantity_area_black,quantity_area_inpaint,quantity_area_cutting_black,"
        f"probability_area_black,probability_area_inpaint,probability_area_cutting_black,"
        f"quantity_insertion_black,quantity_deletion_black,quantity_imd_black,"
        f"quantity_insertion_inpaint,quantity_deletion_inpaint,quantity_imd_inpaint,"
        f"probability_insertion_black,probability_deletion_black,probability_imd_black,"
        f"probability_insertion_inpaint,probability_deletion_inpaint,probability_imd_inpaint\n"
    )

def generate_csv_line_literature(i, method, original_score, literature_results):
    insertion = literature_results[method][5]['auc_insertion']
    deletion = literature_results[method][5]['auc_deletion']
    imd = literature_results[method][5]['imd']

    return (
        f"{i},{method},{original_score:.4f},{literature_results[method][0]:.4f},,"
        f"{literature_results[method][3]:.4f},{literature_results[method][0]:.4f},,"
        f"{literature_results[method][3]:.4f},{literature_results[method][0]:.4f},,"
        f"{literature_results[method][3]:.4f},0.05,{literature_results[method][1]:.4f},"
        f"{literature_results[method][2]:.4f},,{literature_results[method][4]:.4f},"
        f"{literature_results[method][2]:.4f},,{literature_results[method][4]:.4f},"
        f"{literature_results[method][2]:.4f},,{literature_results[method][4]:.4f},"
        f"{insertion:.4f},{deletion:.4f},{imd:.4f},"
        f"0,0,0,"
        f"{insertion:.4f},{deletion:.4f},{imd:.4f},"
        f"0,0,0\n"
    )

def generate_csv_line_methods(i, original_score, methods_results):
    quantity_insertion_black = methods_results['quantity_insertion_deletion_metrics_black']['auc_insertion']
    quantity_deletion_black = methods_results['quantity_insertion_deletion_metrics_black']['auc_deletion']
    quantity_imd_black = methods_results['quantity_insertion_deletion_metrics_black']['imd']
    quantity_insertion_inpaint = methods_results['quantity_insertion_deletion_metrics_inpaint']['auc_insertion']
    quantity_deletion_inpaint = methods_results['quantity_insertion_deletion_metrics_inpaint']['auc_deletion']
    quantity_imd_inpaint = methods_results['quantity_insertion_deletion_metrics_inpaint']['imd']
    probability_insertion_black = methods_results['probability_insertion_deletion_metrics_black']['auc_insertion']
    probability_deletion_black = methods_results['probability_insertion_deletion_metrics_black']['auc_deletion']
    probability_imd_black = methods_results['probability_insertion_deletion_metrics_black']['imd']
    probability_insertion_inpaint = methods_results['probability_insertion_deletion_metrics_inpaint']['auc_insertion']
    probability_deletion_inpaint = methods_results['probability_insertion_deletion_metrics_inpaint']['auc_deletion']
    probability_imd_inpaint = methods_results['probability_insertion_deletion_metrics_inpaint']['imd']

    return (
        f"{i},{methods_results['p_name']},{original_score:.4f},{methods_results['best_score_black']:.4f},"
        f"{methods_results['best_score_inpaint']:.4f},{methods_results['best_score_cutting_black']:.4f},"
        f"{methods_results['quantity_score_black']:.4f},{methods_results['quantity_score_inpaint']:.4f},{methods_results['quantity_score_cutting_black']:.4f},"
        f"{methods_results['probability_score_black']:.4f},{methods_results['probability_score_inpaint']:.4f},{methods_results['probability_score_cutting_black']:.4f},"
        f"{methods_results['problem_time']:.4f},{methods_results['best_area_black']:.4f},{methods_results['best_area_inpaint']:.4f},{methods_results['best_area_cutting_black']:.4f},"
        f"{methods_results['quantity_area_black']:.4f},{methods_results['quantity_area_inpaint']:.4f},{methods_results['quantity_area_cutting_black']:.4f},"
        f"{methods_results['probability_area_black']:.4f},{methods_results['probability_area_inpaint']:.4f},{methods_results['probability_area_cutting_black']:.4f},"
        f"{quantity_insertion_black:.4f},{quantity_deletion_black:.4f},{quantity_imd_black:.4f},"
        f"{quantity_insertion_inpaint:.4f},{quantity_deletion_inpaint:.4f},{quantity_imd_inpaint:.4f},"
        f"{probability_insertion_black:.4f},{probability_deletion_black:.4f},{probability_imd_black:.4f},"
        f"{probability_insertion_inpaint:.4f},{probability_deletion_inpaint:.4f},{probability_imd_inpaint:.4f}\n"
    )
