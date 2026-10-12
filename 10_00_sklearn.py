"""Simple house-price model for the Melbourne housing CSV.

Usage:  python housing_model.py [path/to/melb_data.csv]
"""
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "melb_data.csv"
SEED = 42

df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["Price"])  # need a target for every row

# Features: skip Address/SellerG/Date (high-cardinality or free text) and
# CouncilArea/Suburb (overlap heavily with Regionname/Postcode/coords).
numeric = ["Rooms", "Distance", "Bathroom", "Car", "Landsize",
           "BuildingArea", "YearBuilt", "Lattitude", "Longtitude",
           "Propertycount"]
categorical = ["Type", "Method", "Regionname"]

X = df[numeric + categorical]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

prep = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical),
])

models = {
    "Linear regression": LinearRegression(),
    "Random forest": RandomForestRegressor(
        n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=SEED
    ),
}

print(f"Rows: {len(df)}  (train {len(X_train)}, test {len(X_test)})\n")
print(f"Baseline (predict train median): MAE = "
      f"${mean_absolute_error(y_test, np.full(len(y_test), y_train.median())):,.0f}\n")

fitted = {}
for name, model in models.items():
    pipe = Pipeline([("prep", prep), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    fitted[name] = pipe
    print(f"{name:18s} MAE = ${mean_absolute_error(y_test, pred):>10,.0f}   "
          f"R2 = {r2_score(y_test, pred):.3f}")

# Top feature importances from the random forest
rf = fitted["Random forest"]
names = rf.named_steps["prep"].get_feature_names_out()
imp = pd.Series(rf.named_steps["model"].feature_importances_, index=names)
print("\nTop 8 random-forest features:")
print(imp.sort_values(ascending=False).head(8).round(3).to_string())
