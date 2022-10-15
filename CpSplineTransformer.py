from utils.stats import _weighted_percentile

import numpy as np
import cupy as cp


# This is GPU version
def _interpolate(X, Aug, base_knots, M):
    basis = cp.ones((base_knots.shape[1] - 2) + 2 * M - 1).astype(X.dtype)

    basis[Aug[1:] == Aug[:-1]] = 0

    B = cp.repeat(cp.expand_dims(basis, axis=0), len(X), axis=0)

    B = ((Aug <= X[:, 0][:, cp.newaxis])[:, :-1] & (Aug > X[:, 0][:, cp.newaxis])[:, 1:]) * B

    for i in range(1, M):
        aug1 = Aug[:-i - 1]
        aug2 = Aug[i:-1]
        temp1 = (X - aug1) / (aug2 - aug1)
        temp1[:, aug1 == aug2] = 0

        aug1 = Aug[i + 1:]
        aug2 = Aug[1:-i]
        temp2 = (aug1 - X) / (aug1 - aug2)
        temp2[:, aug1 == aug2] = 0

        B1 = B[:, :-1]
        B2 = B[:, 1:]
        B = temp1 * B1 + temp2 * B2

    return B


class CpSplineTransformer():

    def __init__(self, n_knots=5, degree=3, knots="uniform", extrapolation="constant", fp="float64"):
        self.n_knots = n_knots
        self.degree = degree
        self.knots = knots
        self.M = degree + 1
        self.extrapolation = extrapolation
        self.fp = fp
        self.fitted = False

    def _get_base_knot_positions(self, X, n_knots=10, knots="uniform", sample_weight=None):
        if knots == "quantile":
            percentiles = 100 * np.linspace(start=0, stop=1, num=n_knots, dtype=np.float64)

            if sample_weight is None:
                knots = np.percentile(X, percentiles, axis=0)

            else:
                knots = np.array(
                    [
                        _weighted_percentile(X, sample_weight, percentile)
                        for percentile in percentiles
                    ]
                )

                knots = cp.array(knots)

        else:
            mask = slice(None, None, 1) if sample_weight is None else sample_weight > 0
            x_min = cp.amin(X[mask], axis=0)
            x_max = cp.amax(X[mask], axis=0)

            knots = cp.linspace(
                start=x_min,
                stop=x_max,
                num=n_knots,
                endpoint=True,
                dtype=cp.float64,
            )

        return knots.astype(X.dtype)

    def fit(self, X, sample_weight=None):
        degree = self.degree

        if isinstance(self.fp, str) and self.fp in ["float32", "float64"]:
            if self.fp == "float32":
                self.fp = cp.float32

            else:
                self.fp = cp.float64

        else:
            raise ValueError("fp = [\"float32\", \"float64\"]")

        X = X.astype(self.fp)

        if isinstance(self.knots, str) and self.knots in ["uniform", "quantile", ]:
            base_knots = self._get_base_knot_positions(
                X,
                n_knots=self.n_knots,
                knots=self.knots,
                sample_weight=sample_weight
            )

        elif isinstance(self.knots, cp._core.core.ndarray):
            base_knots = self.knots

        else:
            raise ValueError("knots = [\"uniform\", \"quantile\"] or a ndarray.")

        if self.extrapolation == "periodic":
            period = base_knots[-1] - base_knots[0]
            knots = cp.r_[
                base_knots[-(degree + 1): -1] - period,
                base_knots,
                base_knots[1: (degree + 1)] + period,
            ]

        else:
            dist_min = base_knots[1] - base_knots[0]
            dist_max = base_knots[-1] - base_knots[-2]

            knots = cp.r_[
                cp.linspace(
                    base_knots[0] - degree * dist_min,
                    base_knots[0] - dist_min,
                    num=degree,
                ),
                base_knots,
                cp.linspace(
                    base_knots[-1] + dist_max,
                    base_knots[-1] + degree * dist_max,
                    num=degree,
                ),
            ]

        self.base_knots = base_knots.T
        self.Augment = knots.T
        self.fitted = True

    def transform(self, X):
        if not self.fitted:
            raise Exception("This CpSplineTransformer instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        X = X.astype(self.fp)
        n_samples, n_features = X.shape
        n_splines = self.base_knots.shape[1] + self.degree - 1

        XBS = cp.zeros((n_samples, n_splines * n_features)).astype(X.dtype)

        if self.extrapolation == "error":
            for i in range(n_features):
                temp_X = X[:, i][:, cp.newaxis]
                if cp.any((temp_X > self.base_knots[i, -1]) | (temp_X < self.base_knots[i, 0])):
                    raise ValueError("X contains values beyond the limits of the knots.")

                else:
                    XBS[:, i * n_splines:(i + 1) * n_splines] = _interpolate(temp_X, self.Augment[i], self.base_knots,
                                                                             self.M)

        elif self.extrapolation == "constant":
            for i in range(n_features):
                temp_X = X[:, i][:, cp.newaxis]
                XBS[:, i * n_splines:(i + 1) * n_splines] = _interpolate(temp_X, self.Augment[i], self.base_knots,
                                                                         self.M)

                f_min = _interpolate(cp.array([self.base_knots[i, 0]])[:, cp.newaxis], self.Augment[i], self.base_knots,
                                     self.M)[0]
                f_max = \
                    _interpolate(cp.array([self.base_knots[i, -1]])[:, cp.newaxis], self.Augment[i], self.base_knots,
                                 self.M)[0]
                mask = X[:, i] < self.base_knots[i, 0]
                if cp.any(mask):
                    XBS[mask, (i * n_splines): (i * n_splines + self.degree)] = f_min[
                                                                                :self.degree
                                                                                ]

                mask = X[:, i] > self.base_knots[i, -1]
                if cp.any(mask):
                    XBS[
                    mask,
                    ((i + 1) * n_splines - self.degree): ((i + 1) * n_splines),
                    ] = f_max[-self.degree:]

        elif self.extrapolation == "periodic":
            n_p_splines = self.base_knots.shape[1] - 1
            n = self.Augment.shape[1] - self.degree - 1

            for i in range(n_features):
                x = (self.Augment[i][self.degree] + (X[:, i] - self.Augment[i][self.degree]) % (
                        self.Augment[i][n] - self.Augment[i][self.degree]
                ))[:, cp.newaxis]

                XBS[:, i * n_splines:(i + 1) * n_splines] = _interpolate(x, self.Augment[i], self.base_knots, self.M)

            XBS[:, :n_splines - n_p_splines] = XBS[:, :n_splines - n_p_splines] + XBS[:, n_p_splines:]
            XBS = XBS[:, :n_p_splines]

        elif self.extrapolation == "linear":
            raise Exception("This feature is not yet complete.")

        return XBS

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)