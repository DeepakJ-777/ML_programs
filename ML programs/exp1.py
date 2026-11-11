
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
housing = fetch_california_housing()
X = housing.data[:, [2]]  # AveRooms (already 2D)
y = housing.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Add intercept column
X_train = np.c_[np.ones(len(X_train)), X_train]
X_test = np.c_[np.ones(len(X_test)), X_test]

# 1. Gradient Descent
theta = np.zeros(2)
lr = 0.01
m = len(y_train)

for i in range(1000):
    pred = X_train @ theta
    error = pred - y_train
    gradient = (2/m) * (X_train.T @ error)
    theta -= lr * gradient

gd_pred = X_test @ theta

# 2. Normal Equation
theta_normal = np.linalg.pinv(X_train) @ y_train
ne_pred = X_test @ theta_normal

# Evaluation
print("Gradient Descent")
print("MSE:", mean_squared_error(y_test, gd_pred))
print("R2:", r2_score(y_test, gd_pred))

print("\nNormal Equation")
print("MSE:", mean_squared_error(y_test, ne_pred))
print("R2:", r2_score(y_test, ne_pred))

# Plot regression lines
idx = np.argsort(X_test[:, 1])

plt.scatter(X_test[:, 1], y_test, label="Actual")
plt.plot(X_test[idx, 1], gd_pred[idx], 'r', label="Gradient Descent")
plt.plot(X_test[idx, 1], ne_pred[idx], 'g', label="Normal Equation")

plt.xlabel("Standardized Average Rooms")
plt.ylabel("House Value")
plt.title("Linear Regression")
plt.legend()
plt.grid()
plt.show()
