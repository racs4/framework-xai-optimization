from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np


def add_text_above_image(
    img: Image.Image,
    text: str,
    font_size: int = 20,
    bg_color: str = "white",
    text_color: str = "black",
) -> Image.Image:
    # Criar nova imagem maior (adiciona altura para o texto)
    text_height = font_size + 20  # Altura do texto + padding
    new_width = img.width
    new_height = img.height + text_height

    # Criar imagem expandida
    new_img = Image.new("RGB", (new_width, new_height), bg_color)

    # Colar imagem original na parte inferior
    new_img.paste(img, (0, text_height))

    # Adicionar texto na parte superior
    draw = ImageDraw.Draw(new_img)

    # Tentar usar fonte do sistema (opcional)
    try:
        font = ImageFont.truetype("Arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # Calcular posição centralizada do texto
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    x = (new_width - text_width) // 2
    y = 10

    # Desenhar texto
    draw.text((x, y), text, fill=text_color, font=font)

    return new_img


def clamp(x: int, min_x: int, max_x: int) -> int:
    return max(min_x, min(max_x, x))


def segments_intersect(a, b, c, d):
    def ccw(p, q, r):
        return (r[1] - p[1]) * (q[0] - p[0]) > (q[1] - p[1]) * (r[0] - p[0])

    return (ccw(a, c, d) != ccw(b, c, d)) and (ccw(a, b, c) != ccw(a, b, d))


def polygon_has_crossing_edges(points):
    n = len(points) // 2
    vertices = [(points[2 * i], points[2 * i + 1]) for i in range(n)]
    for i in range(n):
        a1, a2 = vertices[i], vertices[(i + 1) % n]
        for j in range(i + 1, n):
            if abs(i - j) <= 1 or (i == 0 and j == n - 1):
                continue
            b1, b2 = vertices[j], vertices[(j + 1) % n]
            if segments_intersect(a1, a2, b1, b2):
                return True
    return False


def dynamic_polygon_has_crossing_edges(arr):
    n = arr[0]
    return polygon_has_crossing_edges(arr[1 : 2 * n + 1])


def create_folder_if_not_exists(folder_path: str) -> None:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def calcula_perimetros(X):
    num_poligonos = X.shape[0]
    num_pontos = X.shape[1] // 2
    pontos_3d = X.reshape(num_poligonos, num_pontos, 2)
    pontos_seguintes_3d = np.roll(pontos_3d, shift=-1, axis=1)
    diferencas_3d = pontos_seguintes_3d - pontos_3d
    distancias_segmentos = np.linalg.norm(diferencas_3d, axis=2)
    perimetros = np.sum(distancias_segmentos, axis=1)
    return perimetros


def dynamic_calcula_perimetros(X):
    num_poligonos = X.shape[0]
    perimetros = np.zeros(num_poligonos, dtype=float)

    for i in range(num_poligonos):
        num_pontos = int(X[i, 0])
        coords_flat = X[i, 1 : 1 + 2 * num_pontos]
        pontos_2d = coords_flat.reshape(num_pontos, 2)

        pontos_seguintes = np.roll(pontos_2d, shift=-1, axis=0)
        diferencas = pontos_seguintes - pontos_2d
        distancias_segmentos = np.linalg.norm(diferencas, axis=1)
        perimetros[i] = distancias_segmentos.sum()

    return perimetros


# https://github.com/anyoptimization/pymoo/blob/main/pymoo/util/__init__.py
def default_random_state(func_or_seed=None, *, seed=None):
    """
    Decorator that provides a default random state to functions.

    Can be used as:
    - @default_random_state
    - @default_random_state(1)  # with positional seed
    - @default_random_state(seed=1)  # with keyword seed

    If random_state is provided to the function call, it takes precedence.
    """

    def decorator(func, default_seed=None):
        def wrapper(*args, random_state=None, **kwargs):
            if random_state is None:
                # Check if seed is provided in kwargs, otherwise use default_seed
                seed_to_use = kwargs.pop("seed", default_seed)
                random_state = np.random.default_rng(seed_to_use)
            return func(*args, random_state=random_state, **kwargs)

        return wrapper

    # Handle different calling patterns
    if func_or_seed is None:
        # Called as @default_random_state() or @default_random_state(seed=1)
        return lambda func: decorator(func, seed)
    elif callable(func_or_seed):
        # Called as @default_random_state (no parentheses)
        return decorator(func_or_seed, None)
    else:
        # Called as @default_random_state(1) (positional seed)
        return lambda func: decorator(func, func_or_seed)
