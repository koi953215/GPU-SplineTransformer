from GSplineTransformer import GPU_SplineTransformer
import numpy as np

def main():
    n_x = 6
    X = np.arange(n_x).reshape(n_x, 1).astype(np.float64)
    MySP = GPU_SplineTransformer(degree=2, n_knots=3, extrapolation="constant")
    ANS1 = MySP.fit_transform(X)
    print(ANS1)

    return 0

if __name__ == '__main__':
    main()