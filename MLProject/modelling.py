import pandas as pd
import os
import mlflow
import numpy as np
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)

# ✅ Ambil path dataset dari argumen command-line
dataset = os.path.normpath(sys.argv[1].strip('"'))
print("Dataset path:", dataset)

# ✅ Set tracking URI ke folder lokal (CI-safe)
mlflow.set_tracking_uri("file:mlruns")
mlflow.set_experiment("submission model")

# ✅ Mulai MLflow run dengan context manager
with mlflow.start_run() as run:
    print(f"MLFLOW_RUN_ID:{run.info.run_id}")

    # 📥 Load dataset
    df = pd.read_csv(dataset)

    # 🎯 Target dan fitur
    target_column = 'price'
    X = df.drop(columns=target_column)
    y = df[target_column]

    # ✂️ Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 🚀 Model training
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 📊 Evaluasi metrik regresi
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    # 🧾 Logging parameter model
    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("random_state", 42)

    # 📈 Logging metrik evaluasi
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mape", mape)

    # 💾 Simpan model ke MLflow
    example_input = X_test.iloc[:1]
    mlflow.sklearn.log_model(model, artifact_path="model", input_example=example_input)

    # 🖨️ Output hasil evaluasi
    print(f"MSE  : {mse:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAE  : {mae:.2f}")
    print(f"MAPE : {mape * 100:.2f}%")
