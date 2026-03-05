import numpy as np

from pymoo.core.crossover import Crossover
from pymoo.util.misc import crossover_mask
from mestradolib.utils import default_random_state


class PolygonalPointCrossover(Crossover):
    def __init__(self, n_points, **kwargs):
        super().__init__(2, 2, **kwargs)
        self.n_points = n_points

    @default_random_state
    def _do(self, _, X, random_state=None, **kwargs):
        _, n_matings, n_var = X.shape

        valid_cut_positions = [1] + list(range(3, n_var, 2))

        n_valid_positions = len(valid_cut_positions)
        actual_n_points = min(self.n_points, n_valid_positions)

        perms = np.array(
            [random_state.permutation(n_valid_positions) for _ in range(n_matings)]
        )

        selected_indices = perms[:, :actual_n_points]

        r = np.array(valid_cut_positions)[selected_indices]

        r.sort(axis=1)

        r = np.column_stack([r, np.full(n_matings, n_var)])

        M = np.full((n_matings, n_var), False)

        for i in range(n_matings):
            j = 0
            while j < r.shape[1] - 1:
                a, b = r[i, j], r[i, j + 1]
                M[i, a:b] = True
                j += 2

        Xp = crossover_mask(X, M)

        return Xp


class PolygonalSinglePointCrossover(PolygonalPointCrossover):
    def __init__(self, **kwargs):
        super().__init__(n_points=1, **kwargs)
