import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, linear_model, model_selection
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X, y = datasets.load_diabetes(return_X_y=True)

print("Dataset shape:", X.shape)
print("First row:", X[0])

# Use only BMI feature
X = X[:, 2]
X = X.reshape((-1, 1))

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X,
    y,
    test_size=0.2,
)

model = linear_model.LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

sorted_idx = np.argsort(X_test[:, 0])

plt.scatter(X_test, y_test, color="black")
plt.plot(X_test[sorted_idx], y_pred[sorted_idx], color="blue", linewidth=3)
plt.xlabel("Scaled BMI")
plt.ylabel("Disease Progression")
plt.title("Diabetes Progression Against BMI")
plt.savefig("diabetes_bmi_regression.png", dpi=300, bbox_inches="tight")