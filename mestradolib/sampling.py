import numpy as np
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from collections import deque


class CentralSquareSampling(IntegerRandomSampling):
    def _do(self, problem, n_samples, **kwargs):
        # Generate initial random samples
        X = super()._do(problem, n_samples, **kwargs)
        img = problem.img
        center_x, center_y = img.width // 2, img.height // 2
        square_size = min(img.width, img.height) // 4  # Define o tamanho do quadrado central

        # Define os limites do quadrado central
        x_start = max(0, center_x - square_size // 2)
        x_end = min(img.width, center_x + square_size // 2)
        y_start = max(0, center_y - square_size // 2)
        y_end = min(img.height, center_y + square_size // 2)

        # Converte cada amostra para um quadrado central
        for i in range(n_samples):
            mask = np.array(X[i], dtype=bool).reshape(
                (img.height, img.width)
            )
            mask[y_start:y_end, x_start:x_end] = 1
            X[i] = mask.flatten().astype(int)

        return X


class PixelContinuosBlobSampling(IntegerRandomSampling):
    def _do(self, problem, n_samples, **kwargs):
        # Generate initial random samples
        X = super()._do(problem, n_samples, **kwargs)
        img = problem.img
        # Convert each sample to a contiguous blob
        for i in range(n_samples):
            mask = np.array(X[i], dtype=bool).reshape(
                (img.height, img.width)
            )
            mask = make_contiguous_blob(mask)
            X[i] = mask.flatten().astype(int)

        return X


def make_contiguous_blob(mask):

    rows, cols = mask.shape
    visited = np.zeros((rows, cols), dtype=bool)

    # Find the first '1' in the mask to start BFS
    start = None
    for r in range(rows):
        for c in range(cols):
            if mask[r, c] == 1:
                start = (r, c)
                break
        if start:
            break

    if not start:
        return mask  # No '1's in the mask

    # BFS to find all connected '1's
    queue = deque([start])
    visited[start] = True

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

    while queue:
        r, c = queue.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < rows
                and 0 <= nc < cols
                and not visited[nr, nc]
                and mask[nr, nc] == 1
            ):
                visited[nr, nc] = True
                queue.append((nr, nc))

    # Create a new mask with only the contiguous blob
    new_mask = np.zeros_like(mask)
    new_mask[visited] = 1

    return new_mask
