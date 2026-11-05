import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error,r2_score

data=fetch_openml(name="autoMpg",version=1,as_frame=False)
print(data.feature_names)
X=data.data[:,[2]].astype(float)
y=data.target.astype(float)

mask=~np.isnan(X[:,0]) & ~np.isnan(y)
X=X[mask]
y=y[mask]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
linear_model=LinearRegression()
linear_model.fit(X_train,y_train)
linear_pred=linear_model.predict(X_test)
print("linear model evals")
print("mse is",mean_squared_error(y_test,linear_pred))
print("r2 is",r2_score(y_test,linear_pred))

degrees=[2,3,4]
for degree in degrees:
    poly=PolynomialFeatures(degree=degree)
    X_train_poly=poly.fit_transform(X_train)
    X_test_poly=poly.transform(X_test)
    poly_model=LinearRegression()
    poly_model.fit(X_train_poly,y_train)
    poly_pred=poly_model.predict(X_test_poly)
    print(f"poly features degree is {degree}")
    print("mse is",mean_squared_error(y_test,poly_pred))
    print("r2 is",r2_score(y_test,poly_pred))

X_plot=np.linspace(X.min(),X.max(),300).reshape(-1,1)
plt.figure(figsize=(10,6))
plt.scatter(X,y,alpha=0.6,label="data")
plt.plot(X_plot,linear_model.predict(X_plot),color="blue",label="Linear")

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X_train)

    model = LinearRegression()
    model.fit(X_poly, y_train)
    X_plot_poly = poly.transform(X_plot)
    plt.plot(X_plot,model.predict(X_plot_poly),label=f"Polynomial Degree {degree}")

plt.xlabel("horsepower")
plt.ylabel("MPG")
plt.title("linear vs polynomial regression")
plt.legend()
plt.grid(True)
plt.show()
