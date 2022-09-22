from utils.stats import _weighted_percentile
from interpolate import _interpolate
import numpy as np

class GPU_SplineTransformer():

    def __init__(self, n_knots=5, degree=3, knots="uniform", extrapolation="constant"):
        self.n_knots = n_knots
        self.degree = degree
        self.knots = knots
        self.M = degree + 1
        self.extrapolation = extrapolation

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

        else:
            mask = slice(None, None, 1) if sample_weight is None else sample_weight > 0
            x_min = np.amin(X[mask], axis=0)
            x_max = np.amax(X[mask], axis=0)

            knots = np.linspace(
                start=x_min,
                stop=x_max,
                num=n_knots,
                endpoint=True,
                dtype=np.float64,
            )

        return knots

    def fit(self, X, sample_weight=None):
        degree = self.degree

        if isinstance(self.knots, str) and self.knots in ["uniform", "quantile", ]:
            base_knots = self._get_base_knot_positions(
                X,
                n_knots=self.n_knots,
                knots=self.knots,
                sample_weight=sample_weight
            )

        elif isinstance(self.knots, np.ndarray):
            base_knots = self.knots

        else:
            raise ValueError("knots = [\"uniform\", \"quantile\"] or a ndarray.")

        if self.extrapolation == "periodic":
            period = base_knots[-1] - base_knots[0]
            knots = np.r_[
                base_knots[-(degree + 1): -1] - period,
                base_knots,
                base_knots[1: (degree + 1)] + period,
            ]

        else:
            dist_min = base_knots[1] - base_knots[0]
            dist_max = base_knots[-1] - base_knots[-2]

            knots = np.r_[
                np.linspace(
                    base_knots[0] - degree * dist_min,
                    base_knots[0] - dist_min,
                    num=degree,
                ),
                base_knots,
                np.linspace(
                    base_knots[-1] + dist_max,
                    base_knots[-1] + degree * dist_max,
                    num=degree,
                ),
            ]

        self.base_knots = base_knots.T
        self.Augment = knots.T

    def transform(self, X):
        n_samples, n_features = X.shape
        n_splines = self.base_knots.shape[1] + self.degree - 1
        XBS = np.zeros((n_samples, n_splines * n_features))

        if self.extrapolation == "error":
            for i in range(n_features):
                temp_X = X[:, i][:, np.newaxis]
                if np.any((temp_X > self.base_knots[i, -1]) | (temp_X < self.base_knots[i, 0])):
                    raise ValueError("X contains values beyond the limits of the knots.")

                else:
                    XBS[:, i * n_splines:(i + 1) * n_splines] = _interpolate(temp_X, self.Augment[i], self.base_knots,
                                                                             self.M)

        elif self.extrapolation == "constant":
            for i in range(n_features):
                temp_X = X[:, i][:, np.newaxis]
                XBS[:, i * n_splines:(i + 1) * n_splines] = _interpolate(temp_X, self.Augment[i], self.base_knots,
                                                                         self.M)

                f_min = _interpolate(np.array([self.base_knots[i, 0]])[:, np.newaxis], self.Augment[i], self.base_knots,
                                     self.M)[0]
                f_max = \
                _interpolate(np.array([self.base_knots[i, -1]])[:, np.newaxis], self.Augment[i], self.base_knots,
                             self.M)[0]
                mask = X[:, i] < self.base_knots[i, 0]
                if np.any(mask):
                    XBS[mask, (i * n_splines): (i * n_splines + self.degree)] = f_min[
                                                                                :self.degree
                                                                                ]

                mask = X[:, i] > self.base_knots[i, -1]
                if np.any(mask):
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
                ))[:, np.newaxis]

                XBS[:, i * n_splines:(i + 1) * n_splines] = _interpolate(x, self.Augment[i], self.base_knots, self.M)

            XBS[:, :n_splines - n_p_splines] = XBS[:, :n_splines - n_p_splines] + XBS[:, n_p_splines:]
            XBS = XBS[:, :n_p_splines]

        elif self.extrapolation == "linear":
            return

        return XBS

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
