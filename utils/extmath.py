import warnings
import numpy as np

def stable_cumsum(arr, axis=None, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum.
    The following code is from sklearn.
    
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat.
    axis : int, default=None
        Axis along which the cumulative sum is computed.
        The default (None) is to compute the cumsum over the flattened array.
    rtol : float, default=1e-05
        Relative tolerance, see ``np.allclose``.
    atol : float, default=1e-08
        Absolute tolerance, see ``np.allclose``.
    """
    out = np.cumsum(arr, axis=axis, dtype=np.float64)
    expected = np.sum(arr, axis=axis, dtype=np.float64)
    if not np.all(
        np.isclose(
            out.take(-1, axis=axis), expected, rtol=rtol, atol=atol, equal_nan=True
        )
    ):
        warnings.warn(
            "cumsum was found to be unstable: "
            "its last element does not correspond to sum",
            RuntimeWarning,
        )
    return out
