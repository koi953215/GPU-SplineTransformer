import numba as nb
import numpy as np
import numexpr as ne

@nb.jit(nb.float64[:](nb.float64[:], nb.float64[:]), nopython=True, cache=True, parallel=True)
def TEMP1(basis, Aug):
    for i in nb.prange(Aug.shape[0] - 1):
        if (Aug[i] == Aug[i + 1]):
            basis[i] = 0

    return basis


@nb.jit(nb.float64[:, :](nb.float64[:, :], nb.float64[:, :], nb.float64[:]), nopython=True, cache=True, parallel=True)
def TEMP2(X, B, Aug):
    for i in nb.prange(X.shape[0]):
        B[i, :] = ((Aug <= X[i][0])[:-1] & (Aug > X[i][0])[1:]) * B[i, :]

    return B


def TEMP3(X, B, Aug, M):
    for i in range(1, M):
        aug1 = Aug[:-i - 1]
        aug2 = Aug[i:-1]
        temp1 = ne.evaluate("(X - aug1) / (aug2 - aug1)")
        temp1[:, aug1 == aug2] = 0

        aug1 = Aug[i + 1:]
        aug2 = Aug[1:-i]
        temp2 = ne.evaluate("(aug1 - X) / (aug1 - aug2)")
        temp2[:, aug1 == aug2] = 0

        B1 = B[:, :-1]
        B2 = B[:, 1:]
        B = ne.evaluate("temp1 * B1 + temp2 * B2")

    return B


def _interpolate(X, Aug, base_knots, M):
    basis = np.ones((base_knots.shape[1] - 2) + 2 * M - 1)

    basis = TEMP1(basis, Aug)

    B = np.repeat(np.expand_dims(basis, axis=0), len(X), axis=0)

    B = TEMP2(X, B, Aug)

    return TEMP3(X, B, Aug, M)