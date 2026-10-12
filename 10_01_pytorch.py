"""Simple PyTorch neural network for the Melbourne housing CSV.

Usage:  python housing_pytorch.py [path/to/melb_data.csv]
"""
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "melb_data.csv"
torch.manual_seed(42)

# ---------- 1. Data (same features as the scikit-learn script) ----------
df = pd.read_csv(CSV_PATH).dropna(subset=["Price"])
numeric = ["Rooms", "Distance", "Bathroom", "Car", "Landsize",
           "BuildingArea", "YearBuilt", "Lattitude", "Longtitude",
           "Propertycount"]
categorical = ["Type", "Method", "Regionname"]

X_train, X_test, y_train, y_test = train_test_split(
    df[numeric + categorical], df["Price"], test_size=0.2, random_state=42
)

# Neural nets need scaled inputs, so we add StandardScaler after imputing.
prep = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), numeric),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), categorical),
])
Xtr = prep.fit_transform(X_train).astype(np.float32)  # fit on train only
Xte = prep.transform(X_test).astype(np.float32)

# Predict price in millions so the numbers are small and training is stable.
SCALE = 1_000_000
ytr = (y_train.values / SCALE).astype(np.float32).reshape(-1, 1)

Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
Xte_t = torch.from_numpy(Xte)

# ---------- 2. Model: a small feed-forward network ----------
model = nn.Sequential(
    nn.Linear(Xtr.shape[1], 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
)
loss_fn = nn.L1Loss()  # mean absolute error, same metric we report
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------- 3. Training loop ----------
EPOCHS, BATCH = 100, 128
for epoch in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(len(Xtr_t))
    for i in range(0, len(perm), BATCH):
        idx = perm[i:i + BATCH]
        optimizer.zero_grad()                      # clear old gradients
        loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
        loss.backward()                            # compute gradients
        optimizer.step()                           # update weights
    if epoch % 20 == 0:
        print(f"epoch {epoch:3d}  last-batch loss = {loss.item() * SCALE:,.0f}")

# ---------- 4. Evaluate on the held-out test set ----------
model.eval()
with torch.no_grad():
    pred = model(Xte_t).numpy().ravel() * SCALE

print(f"\nPyTorch MLP  MAE = ${mean_absolute_error(y_test, pred):,.0f}   "
      f"R2 = {r2_score(y_test, pred):.3f}")
