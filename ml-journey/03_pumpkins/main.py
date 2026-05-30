from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


DATA_PATH = Path("data/data/US-pumpkins.csv")

pumpkins = pd.read_csv(DATA_PATH)

columns_to_select = [
    "Package",
    "Low Price",
    "High Price",
    "Date",
    "Variety",
    "City Name",
]

pumpkins = pumpkins.loc[:, columns_to_select]

pumpkins = pumpkins[pumpkins["Package"].str.contains("bushel", case=False, na=False)]

price = (pumpkins["Low Price"] + pumpkins["High Price"]) / 2

dates = pd.to_datetime(pumpkins["Date"], errors="coerce")

pumpkins = pumpkins.assign(
    Month=dates.dt.month,
    DayOfYear=dates.apply(
        lambda dt: (dt - datetime(dt.year, 1, 1)).days if pd.notnull(dt) else np.nan
    ),
    Price=price,
)

pumpkins.loc[
    pumpkins["Package"].str.contains("1 1/9", case=False, na=False),
    "Price"
] = pumpkins["Price"] / (1 + 1 / 9)

pumpkins.loc[
    pumpkins["Package"].str.contains("1/2", case=False, na=False),
    "Price"
] = pumpkins["Price"] * 2

pumpkins = pumpkins.dropna()

print("Prepared dataset:")
print(pumpkins.head())
print()
print("Dataset shape:", pumpkins.shape)
print()

print("Correlation:")
print("Month vs Price:", pumpkins["Month"].corr(pumpkins["Price"]))
print("DayOfYear vs Price:", pumpkins["DayOfYear"].corr(pumpkins["Price"]))
print()

pie_pumpkins = pumpkins[pumpkins["Variety"] == "PIE TYPE"].copy()

print("PIE TYPE dataset shape:", pie_pumpkins.shape)
print()

# Simple Linear Regression
X = pie_pumpkins["DayOfYear"].to_numpy().reshape(-1, 1)
y = pie_pumpkins["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=0,
)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

pred = lin_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, pred))
score = lin_reg.score(X_train, y_train)

print("Linear Regression")
print("Coefficient:", lin_reg.coef_[0])
print("Intercept:", lin_reg.intercept_)
print(f"RMSE: {rmse:.3f} ({rmse / np.mean(pred) * 100:.3f}%)")
print("Model determination:", score)
print()

sorted_idx = np.argsort(X_test[:, 0])

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test)
plt.plot(X_test[sorted_idx], pred[sorted_idx])
plt.xlabel("Day of Year")
plt.ylabel("Price")
plt.title("Linear Regression: Pumpkin Price vs Day of Year")
plt.savefig("linear_regression.png", dpi=300, bbox_inches="tight")

# Polynomial Regression
pipeline = make_pipeline(
    PolynomialFeatures(2),
    LinearRegression()
)

pipeline.fit(X_train, y_train)

pred_poly = pipeline.predict(X_test)

rmse_poly = np.sqrt(mean_squared_error(y_test, pred_poly))
score_poly = pipeline.score(X_train, y_train)

print("Polynomial Regression")
print(f"RMSE: {rmse_poly:.3f} ({rmse_poly / np.mean(pred_poly) * 100:.3f}%)")
print("Model determination:", score_poly)
print()

X_range = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1, 1)
y_range = pipeline.predict(X_range)

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test)
plt.plot(X_range, y_range)
plt.xlabel("Day of Year")
plt.ylabel("Price")
plt.title("Polynomial Regression: Pumpkin Price vs Day of Year")
plt.savefig("polynomial_regression.png", dpi=300, bbox_inches="tight")

X_all = (
    pd.get_dummies(pumpkins["Variety"])
    .join(pumpkins["Month"])
    .join(pd.get_dummies(pumpkins["City Name"]))
    .join(pd.get_dummies(pumpkins["Package"]))
)

y_all = pumpkins["Price"]

random_states = [0, 1, 7, 21, 42, 100, 123]

for seed in random_states:
    start_time = time.time()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.33,
        random_state=seed
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

final_pipeline = make_pipeline(
    PolynomialFeatures(2),
    LinearRegression()
)

final_pipeline.fit(X_train, y_train)

final_pred = final_pipeline.predict(X_test)

final_rmse = np.sqrt(mean_squared_error(y_test, final_pred))
final_score = final_pipeline.score(X_train, y_train)

print("All Features Polynomial Regression")
print(f"RMSE: {final_rmse:.3f} ({final_rmse / np.mean(final_pred) * 100:.3f}%)")
print("Model determination:", final_score)