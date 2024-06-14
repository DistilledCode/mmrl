import numpy as np
import chromadb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from itertools import islice
import warnings
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from datetime import datetime
from pathlib import Path
import json
import os

PROJ_DIR = os.environ.get("PROJ_DIR")

os.chdir(PROJ_DIR)

warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)

INCLUDE_METADATA = False


history = Path("../models/history.json")


def get_processed_dfs(path):
    if not path.exists():
        path.touch()
        print(f"File '{path}' created.")
        proc_dfs = []
    else:
        print(f"[*] File '{path}' already exists. Loading it.")
        with open(path, "r") as f:
            proc_dfs = json.load(f)
        print(f"[*] Already have processed {len(proc_dfs)} dfs")
    return proc_dfs


def batched(iterable, n):
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch


print("[*] Loading embeddings.")

imagedf = pd.read_parquet("../dump/final_images.parquet")
chroma_client = chromadb.PersistentClient(path="../embeddings")
features_collection_1 = chroma_client.get_collection(name="features_1")
features_collection_2 = chroma_client.get_collection(name="features_2")

if INCLUDE_METADATA:
    collection_data = features_collection_2.get(include=["embeddings"])
else:
    collection_data = features_collection_1.get(include=["embeddings"])
data = collection_data["ids"]
print("[*] Loading popularity scores.")
imagedf["post_id"] = pd.Categorical(imagedf["post_id"], categories=data, ordered=True)
y = (
    imagedf[imagedf["post_id"].isin(data)]
    .sort_values("post_id")
    .reset_index(drop=True)["popularity"]
)

X = np.array(collection_data["embeddings"])
y = np.array(y)

print("[*] Shuffling data.")
X, y = shuffle(X, y, random_state=42)

print("[*] Creating test/train split.")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print(
    f"{X_train.shape=}",
    f"{X_train[0].shape=}",
    f"{X_test.shape=}",
    f"{X_test[0].shape=}",
    f"{y_train.shape=}",
    f"{y_test.shape=}",
    sep="\n",
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
)

INPUT = 4096 if INCLUDE_METADATA is False else 4112

model = Sequential(
    [
        Dense(
            2048,
            input_shape=(INPUT,),
            kernel_regularizer=regularizers.l2(0.0002),
        ),
        BatchNormalization(),
        ReLU(),
        Dense(512, kernel_regularizer=regularizers.l2(0.0002)),
        BatchNormalization(),
        ReLU(),
        Dense(256, kernel_regularizer=regularizers.l2(0.0002)),
        BatchNormalization(),
        ReLU(),
        Dense(128, kernel_regularizer=regularizers.l2(0.0002)),
        BatchNormalization(),
        ReLU(),
        Dense(1),
    ]
)
initial_learning_rate = 0.0001
lr_schedule = ExponentialDecay(
    initial_learning_rate,
    decay_steps=1,  # decay every epoch
    decay_rate=0.4,
    staircase=True,
)

# Define the optimizer with the learning rate schedule
optimizer = Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss="mean_squared_error")

# Print the model summary
model.summary()

model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
)


#! https://www.perplexity.ai/search/Compare-the-two-RwaSdcf2TleDNHeEFxzWKg


loss = model.evaluate(X_test, y_test)

y_pred = model.predict(X_test)
print(f"Test Loss: {loss}")

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)

mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae}")

mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse}")

rmse = np.sqrt(mse)
print(f"RMSE: {rmse}")

r2 = r2_score(y_test, y_pred)
print(f"R²: {r2}")

mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"MAPE: {mape * 100}%")

print("[*] Saving Model")

save_path = f'../models/{datetime.now().strftime("%y_%m_%d_%H_%m")}-{len(y)}.keras'
model.save(save_path)

metrics = {
    save_path: {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
    }
}


past_metrics = get_processed_dfs(history)
past_metrics.append(metrics)

print("[*] Saving Model Metrics")
with open("../models/history.json", "w") as f:
    json.dump(past_metrics, f)
