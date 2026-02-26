import numpy as np

def split(X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2, seed: int = 42):

    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_ratio)

    test_idx  = indices[:test_size]
    train_idx = indices[test_size:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]