import os
import numpy as np
import matplotlib.pyplot as plt
from src.utils import ensure_dir

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot

def plot_actual_vs_pred(y_true, y_pred, path: str):
    ensure_dir(os.path.dirname(path))
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.4)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted Prices")

    min_, max_ = y_true.min(), y_true.max()
    plt.plot([min_, max_], [min_, max_], 'r--')
    plt.savefig(path); plt.close()

def plot_error_distribution(y_true, y_pred, path: str):
    ensure_dir(os.path.dirname(path))
    errors = y_true - y_pred
    plt.figure()
    plt.hist(errors, bins=50, alpha=0.7, color="tomato")
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.title("Error Distribution")
    plt.savefig(path); plt.close()