import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

data = load_diabetes()

X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)

print("Linear Regression")
print("MSE:", mean_squared_error(y_test, linear_pred))
print("R2:", r2_score(y_test, linear_pred))



ridge_model = RidgeCV(alphas=[0.01,0.1,1,10,100], cv=5)

ridge_model.fit(X_train, y_train)

ridge_pred = ridge_model.predict(X_test)

print("\nRidge Regression")
print("Best Alpha:", ridge_model.alpha_)
print("MSE:", mean_squared_error(y_test, ridge_pred))
print("R2:", r2_score(y_test, ridge_pred))


lasso_model = LassoCV(alphas=[0.01,0.1,1,10,100], cv=5, max_iter=100000)
lasso_model.fit(X_train, y_train)
lasso_pred = lasso_model.predict(X_test)

print("\nLasso Regression")
print("Best Alpha:", lasso_model.alpha_)
print("MSE:", mean_squared_error(y_test, lasso_pred))
print("R2:", r2_score(y_test, lasso_pred))
