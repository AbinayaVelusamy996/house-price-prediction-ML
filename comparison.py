
import time, numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from src.train_test_split import split
from src.evaluation       import mse, rmse, r2_score

df = pd.read_csv("data/processed_data.csv")
X  = df.drop(columns=["price"]).values
y  = df["price"].values

X_tr, X_te, y_tr, y_te = split(X, y, test_ratio=0.20, seed=42)

t0 = time.time()
sk_model = LinearRegression(fit_intercept=True)
sk_model.fit(X_tr, y_tr)
sk_time = time.time() - t0
y_pred_sk = sk_model.predict(X_te)


w = np.load("outputs/models/weights.npy")
b = np.load("outputs/models/bias.npy")[0]
y_pred_np = X_te.dot(w) + b

tbl = pd.DataFrame({
    "Implementation" : ["NumPy-GD", "sk-learn OLS"],
    "Train time (s)": [float(np.load("outputs/models/train_time.npy")[0]), sk_time],
    "MSE"            : [mse(y_te, y_pred_np),  mse(y_te, y_pred_sk)],
    "RMSE"           : [rmse(y_te, y_pred_np), rmse(y_te, y_pred_sk)],
    "R²"             : [r2_score(y_te, y_pred_np), r2_score(y_te, y_pred_sk)]
})
print("\n====  Benchmark: From-Scratch vs scikit-learn  ====\n")
print(tbl.to_string(index=False, float_format=lambda x: f'{x:,.3f}'))