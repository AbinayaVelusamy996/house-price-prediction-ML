import numpy as np

class MinMaxScaler:

    def __init__(self):
        self.min_ = None
        self.range_ = None

    def fit(self, X: np.ndarray):
        self.min_   = X.min(axis=0)
        self.range_ = X.max(axis=0) - self.min_

        self.range_[self.range_ == 0] = 1.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.min_) / self.range_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)