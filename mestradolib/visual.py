import numpy as np
from mestradolib.model import ModelWrapper, get_img_score
from mestradolib.problems import EvaluationObject, minimize_rectangle_problem
from mestradolib.mask import mask_from_integers
from PIL import Image, ImageDraw, ImageFont
from pymoo.core.result import Result
from fastai.learner import Learner
import matplotlib.pyplot as plt
import cv2


def show_image(img: Image.Image, title: str = "Image") -> None:
    plt.figure(figsize=(5, 5))
    plt.imshow(np.array(img))
    plt.title(title)
    plt.axis("off")
    plt.show()


def get_masked_image_and_show(
    img: Image.Image,
    res: Result,
    threshold: float,
    create_heatmap_function: callable,
    apply_heatmap_function: callable,
) -> None:
    img_res, _ = get_masked_image(
        img, res, threshold, create_heatmap_function, apply_heatmap_function
    )
    show_image(img_res, title="Masked Image")


def add_title_to_image(
    img: Image.Image, title: str, font_size=None, font_pos=(10, 5)
) -> Image.Image:
    # Create a new image with extra space for the title
    new_img = Image.new("RGB", (img.width, img.height + 30), "white")
    new_img.paste(img, (0, 30))

    # Load a font (replace 'arial.ttf' with a valid font file on your system)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()  # Fallback to default font if not found

    # Draw the title on the new image
    draw = ImageDraw.Draw(new_img)
    draw.multiline_text(font_pos, title, fill="black", font=font)

    return new_img


def get_masked_image(
    img: Image.Image,
    res: Result,
    threshold: float,
    create_heatmap_function: callable,
    apply_heatmap_function: callable,
    title: str | None = None,
    font_size=None,
    font_pos=(10, 5),
) -> tuple[Image.Image, float]:
    heatmap = create_heatmap_function(img, res)

    max_value = np.max(heatmap)
    min_value = np.min(heatmap)

    range_value = max_value - min_value

    threshold_value = min_value + (range_value * threshold)

    heatmap_solutions = (heatmap >= threshold_value).astype(np.int32)

    img_res, area = apply_heatmap_function(img, heatmap_solutions)

    if title is not None:
        img_res = add_title_to_image(img_res, title, font_size, font_pos)

    return img_res, area


def get_masked_image_with_score(
    img: Image.Image,
    res: Result,
    threshold: float,
    create_heatmap_function: callable,
    apply_heatmap_function: callable,
    model_wrapper: ModelWrapper,
    original_score: float,
    original_idx: int,
    font_size=None,
    font_pos=(10, 5),
) -> Image.Image:
    masked_image, area = get_masked_image(
        img, res, threshold, create_heatmap_function, apply_heatmap_function
    )
    masked_image_score, _ = model_wrapper.get_img_score(masked_image, original_idx)
    return (
        add_title_to_image(
            masked_image,
            f"{masked_image_score:.4f}\n{original_score:.4f}",
            font_size,
            font_pos,
        ),
        masked_image_score,
        area,
    )


def minimize_rectangle_problem_and_get_masked_image(
    img: Image.Image,
    learner: Learner,
    evaluation_object: EvaluationObject,
    threshold: float,
    create_heatmap_function,
    apply_heatmap_function,
) -> Image.Image:
    res = minimize_rectangle_problem(img, evaluation_object, learner)
    masked_image, _ = get_masked_image(
        img, res, threshold, create_heatmap_function, apply_heatmap_function
    )
    return masked_image


def show_rectangle(img: Image.Image, rect: list[int]) -> Image.Image:
    x1, y1, x2, y2 = rect
    # copy img
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    return img_copy


def save_heatmap_result(
    img: Image.Image,
    res: Result,
    create_heatmap_function: callable,
    save_path: str,
    title: str = "Heatmap of Non-dominated Solutions",
    f: callable = None,
) -> None:
    heatmap_solutions = create_heatmap_function(img, res)
    plt.figure(figsize=(5, 5))
    plt.imshow(np.array(img), cmap="gray", alpha=0.7)
    plt.imshow(heatmap_solutions, cmap="hot_r", alpha=0.5)
    plt.title(title)
    plt.colorbar(label="Number of Solutions")
    plt.savefig(save_path)
    plt.close()


def show_any_heatmap_result(
    img: Image.Image,
    res: Result,
    create_heatmap_function: callable,
    title: str = "Heatmap of Non-dominated Solutions",
    f: callable = None,
) -> None:
    heatmap_solutions = create_heatmap_function(img, res, f)
    plt.figure(figsize=(5, 5))
    plt.imshow(np.array(img), cmap="gray", alpha=0.7)
    plt.imshow(heatmap_solutions, cmap="hot_r", alpha=0.5)
    plt.title(title)
    plt.colorbar(label="Number of Solutions")
    plt.show()


def create_rectangle_heatmap(
    img: Image.Image, res: Result, f: callable = None
) -> np.ndarray:
    # Create a blank heatmap
    heatmap_solutions = np.zeros((img.height, img.width), dtype=np.int32)

    # Mark rectangles from non-dominated solutions
    for solution in res.X:  # type: ignore
        if f is not None:
            f(solution, heatmap_solutions)
        else:
            x1, y1, x2, y2 = [coord for coord in solution]
            heatmap_solutions[y1:y2, x1:x2] += 1

    return heatmap_solutions


def create_rectangle_best_image(
    img: Image.Image, res: Result, f: callable = None
) -> np.ndarray:
    # Create a blank heatmap
    heatmap_solutions = np.zeros((img.height, img.width), dtype=np.int32)

    # Mark rectangles from best solution
    # get image with highest first objective value
    solution = res.X[np.argmin(res.F[:, 0])]
    if f is not None:
        f(solution, heatmap_solutions)
    else:
        x1, y1, x2, y2 = [coord for coord in solution]
        heatmap_solutions[y1:y2, x1:x2] += 1

    return heatmap_solutions


def show_heatmap_result(img: Image.Image, res: Result, f: callable = None) -> None:
    show_any_heatmap_result(img, res, create_rectangle_heatmap, f)


def create_rectangle_probability_heatmap(
    img: Image.Image, res: Result, f: callable = None
) -> np.ndarray:
    # Create a blank heatmap
    heatmap_solutions = np.zeros((img.height, img.width), dtype=np.float64)

    # Mark rectangles from non-dominated solutions
    for i in range(len(res.X)):
        solution = res.X[i]
        objective = res.F[i]
        if f is not None:
            f(solution, objective, heatmap_solutions)
        else:
            x1, y1, x2, y2 = [coord for coord in solution]
            heatmap_solutions[y1:y2, x1:x2] += -objective[0]

    return heatmap_solutions


def show_heatmap_probability_result(
    img: Image.Image, res: Result, f: callable = None
) -> None:
    show_any_heatmap_result(img, res, create_rectangle_probability_heatmap, f)


def create_polygonal_heatmap(
    img: Image.Image, res: Result, f: callable = None, f_extract_polygon=None
) -> np.ndarray:
    # Create a blank heatmap
    heatmap_solutions = np.zeros((img.height, img.width), dtype=np.int32)

    # Mark polygons from non-dominated solutions
    for solution in res.X:  # type: ignore
        if f is not None:
            f(solution, heatmap_solutions)
        else:
            f_extract_polygon = (
                extract_static_polygon
                if f_extract_polygon is None
                else f_extract_polygon
            )
            mask = Image.new("L", (img.width, img.height), 0)
            polygon = f_extract_polygon(solution)
            ImageDraw.Draw(mask).polygon(polygon, outline=1, fill=1)
            heatmap_solutions += np.array(mask)

    return heatmap_solutions


def create_polygonal_best_image(
    img: Image.Image, res: Result, f: callable = None, f_extract_polygon=None
) -> np.ndarray:
    # Create a blank heatmap
    heatmap_solutions = np.zeros((img.height, img.width), dtype=np.int32)

    # Mark polygons from best solution
    # get image with highest first objective value
    solution = res.X[np.argmin(res.F[:, 0])]
    if f is not None:
        f(solution, heatmap_solutions)
    else:
        f_extract_polygon = (
            extract_static_polygon if f_extract_polygon is None else f_extract_polygon
        )
        mask = Image.new("L", (img.width, img.height), 0)
        polygon = f_extract_polygon(solution)
        ImageDraw.Draw(mask).polygon(polygon, outline=1, fill=1)
        heatmap_solutions += np.array(mask)

    return heatmap_solutions


def show_polygonal_heatmap(img: Image.Image, res: Result, f: callable = None) -> None:
    show_any_heatmap_result(img, res, create_polygonal_heatmap, f)


def create_polygonal_probability_heatmap(
    img: Image.Image, res: Result, f: callable = None, f_extract_polygon=None
) -> np.ndarray:
    # Create a blank heatmap
    heatmap_solutions = np.zeros((img.height, img.width), dtype=np.float64)

    # If res.X is one dimension, transforms in two
    if res.X.ndim == 1:
        res.X = np.expand_dims(res.X, axis=0)
    # If res.F is one dimension, transforms in two
    if res.F.ndim == 1:
        res.F = np.expand_dims(res.F, axis=0)

    # Mark polygons from non-dominated solutions
    for i in range(len(res.X)):  # type: ignore
        solution = res.X[i]
        objective = res.F[i]
        if f is not None:
            f(solution, heatmap_solutions)
        else:
            f_extract_polygon = (
                extract_static_polygon
                if f_extract_polygon is None
                else f_extract_polygon
            )
            mask = Image.new("L", (img.width, img.height), 0)
            polygon = f_extract_polygon(solution)
            ImageDraw.Draw(mask).polygon(polygon, outline=1, fill=1)
            heatmap_solutions += np.array(mask) * (-objective[0])

    return heatmap_solutions


def show_polygonal_probability_heatmap(
    img: Image.Image, res: Result, f: callable = None
) -> None:
    show_any_heatmap_result(img, res, create_polygonal_probability_heatmap, f)


def save_pareto_front(res: Result, file_path: str) -> None:
    plt.figure(figsize=(8, 2))
    plt.scatter(res.F[:, 0], res.F[:, 1], c="blue", label="Non-dominated Solutions")
    plt.xlabel("Diff prob")
    plt.ylabel("Area/Perimetro")
    plt.title("Pareto Front")
    plt.legend()
    plt.grid()
    plt.savefig(file_path)
    plt.close()


def show_pareto_front(
    res: Result, label1: str = "Diff prob", label2: str = "Area/Perimetro"
) -> None:
    plt.figure(figsize=(8, 2))
    plt.scatter(res.F[:, 0], res.F[:, 1], c="blue", label="Non-dominated Solutions")
    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.title("Pareto Front")
    plt.legend()
    plt.grid()
    plt.show()


def create_mask_best_image(
    img: Image.Image, res: Result, f: callable = None
) -> np.ndarray:
    # Create a blank heatmap
    heatmap_solutions = np.zeros((img.height, img.width), dtype=np.int32)

    # Mark pixels from best solution
    # get image with highest first objective value
    solution = res.X

    # Se tiver mais que uma solucao, pega uma
    if res.X.ndim == 2:
        solution = res.X[0]

    if f is not None:
        f(solution, heatmap_solutions)
    else:
        mask = mask_from_integers(solution, img.size).astype(bool)
        heatmap_solutions += mask

    return heatmap_solutions


def create_mask_heatmap(
    img: Image.Image, res: Result, f: callable = None
) -> np.ndarray:
    # Create a blank heatmap
    heatmap_solutions = np.zeros((img.height, img.width), dtype=np.int32)

    # Se tiver mais que uma solucao, pega uma
    if res.X.ndim == 2:
        res.X = res.X[0]

    # Mark pixels from non-dominated solutions
    for solution in [res.X]:  # type: ignore
        if f is not None:
            f(solution, heatmap_solutions)
        else:
            mask = mask_from_integers(solution, img.size).astype(bool)
            heatmap_solutions += mask

    return heatmap_solutions


def show_mask_heatmap(img: Image.Image, res: Result, f: callable = None) -> None:
    show_any_heatmap_result(img, res, create_mask_heatmap, f)


def create_mask_probability_heatmap(
    img: Image.Image, res: Result, f: callable = None
) -> np.ndarray:
    # Create a blank heatmap
    heatmap_solutions = np.zeros((img.height, img.width), dtype=np.float64)

    # Se tiver mais que uma solucao, pega uma
    if res.X.ndim == 2:
        res.X = res.X[0]
        res.F = res.F[0]

    # Mark pixels from non-dominated solutions
    for i in range(len([res.X])):  # type: ignore
        solution = [res.X][i]
        objective = [res.F][i]
        if f is not None:
            f(solution, heatmap_solutions)
        else:
            mask = mask_from_integers(solution, img.size).astype(bool)
            heatmap_solutions += mask * (-objective[0])

    return heatmap_solutions


def show_mask_probability_heatmap(
    img: Image.Image, res: Result, f: callable = None
) -> None:
    show_any_heatmap_result(img, res, create_mask_probability_heatmap, f)


def apply_black_heatmap(
    img: Image.Image, heatmap_solutions: np.ndarray
) -> tuple[Image.Image, float]:
    # Create a copy of the original image to avoid modifying it
    img_copy = img.copy()
    img_array = np.array(img_copy)

    # Create a mask where heatmap_solutions is greater than 0
    mask = heatmap_solutions > 0

    # Set the pixels in the original image to black where the mask is True
    img_array[mask] = [0, 0, 0]

    # Convert back to PIL Image
    img_result = Image.fromarray(img_array)
    # Calculate the total area of the masked region
    total_area = np.sum(mask) / (mask.shape[0] * mask.shape[1])
    return img_result, total_area


def apply_cutting_heatmap(
    img: Image.Image, heatmap_solutions: np.ndarray
) -> tuple[Image.Image, float]:
    # Create a copy of the original image to avoid modifying it
    img_copy = img.copy()
    img_array = np.array(img_copy)

    # Create a mask where heatmap_solutions is greater than 0
    mask = heatmap_solutions > 0

    # Set the pixels in the original image to black where the mask is False
    img_array[~mask] = [0, 0, 0]

    # Convert back to PIL Image
    img_result = Image.fromarray(img_array)
    # Calculate the total area of the masked region
    total_area = np.sum(mask) / (mask.shape[0] * mask.shape[1])
    return img_result, total_area


def apply_inpainting_heatmap(
    img: Image.Image, heatmap_solutions: np.ndarray
) -> tuple[Image.Image, float]:
    # Convert PIL image to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Create a mask where heatmap_solutions is greater than 0
    mask = (heatmap_solutions > 0).astype(np.uint8) * 255

    # Inpaint the image using the mask
    inpainted_img_cv = cv2.inpaint(
        img_cv, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA
    )

    # Convert back to PIL Image
    inpainted_img = Image.fromarray(cv2.cvtColor(inpainted_img_cv, cv2.COLOR_BGR2RGB))
    # Calculate the total area of the inpainted region
    total_area = np.sum(mask > 0)
    return inpainted_img, total_area


def extract_static_polygon(solution: np.ndarray) -> list[tuple]:
    return [(solution[i], solution[i + 1]) for i in range(0, len(solution), 2)]


def extract_dynamic_polygon(solution: np.ndarray) -> list[tuple]:
    n = int(solution[0])
    coords = solution[1 : 2 * n + 1]
    return [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
