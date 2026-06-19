import pandas as pd
import numpy as np
import mlflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os

# Load Data
print("Loading data...")
X_train_2d = pd.read_csv("inflow_preprocessing/X_train.csv").values
y_train_2d = pd.read_csv("inflow_preprocessing/y_train.csv").values
X_test_2d = pd.read_csv("inflow_preprocessing/X_test.csv").values
y_test_2d = pd.read_csv("inflow_preprocessing/y_test.csv").values

look_back = 168
n_features = 4
output_steps = 168

X_train = X_train_2d.reshape((X_train_2d.shape[0], look_back, n_features))
y_train = y_train_2d.reshape((y_train_2d.shape[0], output_steps, 1))
X_test = X_test_2d.reshape((X_test_2d.shape[0], look_back, n_features))
y_test = y_test_2d.reshape((y_test_2d.shape[0], output_steps, 1))

with mlflow.start_run():
    mlflow.tensorflow.autolog()

    model = Sequential([
        LSTM(8, activation='relu', input_shape=(look_back, n_features)),
        Dense(output_steps)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # Train with 1 epoch and small subset just for CI to pass quickly
    print("Training model for CI...")
    model.fit(X_train[:100], y_train[:100], epochs=1, batch_size=32)

    # Save model explicitly so workflow can find it and use our custom conda.yaml
    mlflow.tensorflow.log_model(model, "lstm_model", conda_env="conda.yaml")
    print("Training finished.")
