from pymoo.core.mutation import Mutation
from pymoo.core.variable import get, Real
from pymoo.operators.repair.to_bound import set_to_bounds_if_outside
import numpy as np
from mestradolib.utils import default_random_state


class PixelContinuosMutation(Mutation):
    def _do(self, problem, X, **kwargs):
        X_mutated = X.copy()
        for i in range(X.shape[0]):
            idx = np.where(X[i] == 1)[0]
            if len(idx) == 0:
                continue
            idx_to_mutate = np.random.choice(idx)
            X_mutated[i, idx_to_mutate] = 0

            neighbors = self.get_neighbors(idx_to_mutate, problem.n_var)
            if len(neighbors) > 0:
                new_idx = np.random.choice(neighbors)
                X_mutated[i, new_idx] = 1

        return X_mutated

    def get_neighbors(self, index, n_var):
        neighbors = []
        row_length = int(np.sqrt(n_var))
        col_length = row_length

        row = index // row_length
        col = index % col_length

        for r in range(max(0, row - 1), min(row + 2, row_length)):
            for c in range(max(0, col - 1), min(col + 2, col_length)):
                if r == row and c == col:
                    continue
                neighbors.append(r * row_length + c)

        return neighbors


# code modified from https://github.com/anyoptimization/pymoo/blob/main/pymoo/operators/mutation/pm.py
@default_random_state
def polygonal_mut_pm(X, xl, xu, eta, prob, at_least_once, random_state=None):
    n, n_var = X.shape
    assert len(eta) == n
    assert len(prob) == n

    Xp = np.full(X.shape, np.inf)

    mut = polygonal_mut_binomial(
        n, n_var, prob, at_least_once=at_least_once, random_state=random_state
    )
    mut[:, xl == xu] = False

    Xp[:, :] = X

    _xl = np.repeat(xl[None, :], X.shape[0], axis=0)[mut]
    _xu = np.repeat(xu[None, :], X.shape[0], axis=0)[mut]

    X = X[mut]
    eta = np.tile(eta[:, None], (1, n_var))[mut]

    delta1 = (X - _xl) / (_xu - _xl)
    delta2 = (_xu - X) / (_xu - _xl)

    mut_pow = 1.0 / (eta + 1.0)

    rand = random_state.random(X.shape)
    mask = rand <= 0.5
    mask_not = np.logical_not(mask)

    deltaq = np.zeros(X.shape)

    xy = 1.0 - delta1
    val = 2.0 * rand + (1.0 - 2.0 * rand) * (np.power(xy, (eta + 1.0)))
    d = np.power(val, mut_pow) - 1.0
    deltaq[mask] = d[mask]

    xy = 1.0 - delta2
    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (np.power(xy, (eta + 1.0)))
    d = 1.0 - (np.power(val, mut_pow))
    deltaq[mask_not] = d[mask_not]

    # mutated values
    _Y = X + deltaq * (_xu - _xl)

    # back in bounds if necessary (floating point issues)
    _Y[_Y < _xl] = _xl[_Y < _xl]
    _Y[_Y > _xu] = _xu[_Y > _xu]

    # set the values for output
    Xp[mut] = _Y

    # in case out of bounds repair (very unlikely)
    Xp = set_to_bounds_if_outside(Xp, xl, xu)

    return Xp


def polygonal_mut_binomial(n, m, prob, at_least_once=True, random_state=None):
    prob = np.ones(n) * prob

    M_primeira = random_state.random(n) < prob

    num_coords = m - 1
    num_pontos = num_coords // 2

    M_pontos = random_state.random((n, num_pontos)) < prob[:, None]

    M_expanded = np.repeat(M_pontos, 2, axis=1)

    M = np.zeros((n, m), dtype=bool)

    M[:, 0] = M_primeira

    end_idx = 1 + num_pontos * 2
    M[:, 1:end_idx] = M_expanded

    return M


class DynamicPolygonMutation(Mutation):
    def __init__(self, prob=0.9, eta=20, at_least_once=False, **kwargs):
        super().__init__(prob=prob, **kwargs)
        self.at_least_once = at_least_once
        self.eta = Real(eta, bounds=(3.0, 30.0), strict=(1.0, 100.0))

    def _do(self, problem, X, params=None, *args, random_state=None, **kwargs):
        X = X.astype(float)

        eta = get(self.eta, size=len(X))
        prob_var = self.get_prob_var(problem, size=len(X))
        np.set_printoptions(threshold=np.inf)

        Xp = polygonal_mut_pm(
            X,
            problem.xl,
            problem.xu,
            eta,
            prob_var,
            at_least_once=self.at_least_once,
            random_state=random_state,
        )

        return Xp
