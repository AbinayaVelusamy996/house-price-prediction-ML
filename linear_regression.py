import os
import numpy as np
import matplotlib.pyplot as plt
from src.utils import ensure_dir

class LinearRegressionGD:

    def __init__(self, lr=0.01, epochs=1000, verbose=True):
        self.lr      = lr
        self.epochs  = epochs
        self.verbose = verbose
        self.w       = None
        self.b       = None
        self.losses  = []

    @staticmethod
    def _mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        for epoch in range(1, self.epochs + 1):
            y_pred = X.dot(self.w) + self.b
            error  = y_pred - y


            dw = (2 / n_samples) * X.T.dot(error)
            db = (2 / n_samples) * np.sum(error)


            self.w -= self.lr * dw
            self.b -= self.lr * db

            loss = self._mse(y, y_pred)
            self.losses.append(loss)

            if self.verbose and epoch % 100 == 0:
                print(f"[GD] epoch {epoch:4d}/{self.epochs}, MSE={loss:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X.dot(self.w) + self.b

    def save_weights(self, out_dir: str):
        ensure_dir(out_dir)
        np.save(f"{out_dir}/weights.npy", self.w)
        np.save(f"{out_dir}/bias.npy",    np.array([self.b]))

    def plot_loss(self, path: str):
        ensure_dir(os.path.dirname(path))
        plt.figure()
        plt.plot(self.losses)
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Loss vs Epoch")
        plt.savefig(path)
        plt.close()