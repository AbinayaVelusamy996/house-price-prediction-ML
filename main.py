
import argparse
import time
import numpy as np
import pandas as pd

from src.data_ingestion      import ingest
from src.data_cleaning       import clean
from src.data_validation     import validate_ranges
from src.feature_engineering import add_features
from src.normalization       import MinMaxScaler
from src.train_test_split    import split
from src.linear_regression   import LinearRegressionGD
from src.evaluation          import mse, rmse, r2_score, \
                                     plot_actual_vs_pred, plot_error_distribution
from src.utils               import save_csv

DATA_DIR    = "data"
OUTPUT_FIGS = "outputs/figures"
OUTPUT_MODEL= "outputs/models"

def run_pipeline(raw_path: str):

    df = ingest(raw_path, f"{DATA_DIR}/raw_data.csv")


    df = clean(df, f"{DATA_DIR}/cleaned_data.csv")


    df = validate_ranges(df)


    df = add_features(df)


    target_col = "price"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].values.astype(float)
    y = df[target_col].values.astype(float)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    processed_df = pd.DataFrame(X_scaled, columns=feature_cols)
    processed_df[target_col] = y
    save_csv(processed_df, f"{DATA_DIR}/processed_data.csv")


    X_train, X_test, y_train, y_test = split(X_scaled, y, test_ratio=0.2)


    model = LinearRegressionGD(lr=0.05, epochs=1500, verbose=True)
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    np.save("outputs/models/train_time.npy", np.array([train_time]))
    model.save_weights(OUTPUT_MODEL)
    model.plot_loss(f"{OUTPUT_FIGS}/loss_curve.png")


    y_pred = model.predict(X_test)
    metrics = {
        "MSE" : mse(y_test, y_pred),
        "RMSE": rmse(y_test, y_pred),
        "R2"  : r2_score(y_test, y_pred),
        "Training Time (s)" : train_time
    }
    print("\n--- Evaluation on Test set ---")
    for k,v in metrics.items():
        print(f"{k:>15s}: {v:.4f}")


    plot_actual_vs_pred(y_test, y_pred, f"{OUTPUT_FIGS}/actual_vs_pred.png")
    plot_error_distribution(y_test, y_pred, f"{OUTPUT_FIGS}/error_dist.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to Kaggle house csv")
    args = parser.parse_args()
    run_pipeline(args.data)