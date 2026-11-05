import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing=fetch_california_housing()

X=housing.data[:,2]
y=housing.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

X_train=X_train.reshape(-1,1)
X_test=X_test.reshape(-1,1)

scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

X_train=np.c_[np.ones(len(X_train)),X_train]
X_test=np.c_[np.ones(len(X_test)),X_test]
theta=np.zeros(2)

iterations=1000
learning_rate=0.01
m=len(y_train)

for i in range(iterations):
  prediction=X_train@theta
  errors=prediction-y_train
  gradient=(2/m)*(X_train.T@errors)
  theta=theta-learning_rate*gradient

y_pred=X_test@theta
print("Gradient equation")
print("mse is ",mean_squared_error(y_pred,y_test))
print("r2 score is",r2_score(y_test,y_pred))
theta_normal=np.linalg.pinv(X_train)@y_train
n_predictions=X_test@theta_normal

print("normal equation")
print("mse is ",mean_squared_error(n_predictions,y_test))
print("r2 is",r2_score(y_test,n_predictions))
sorted_indices=np.argsort(X_test[:,1])
plt.figure(figsize=(8,5))
plt.scatter(X_test[:,1],y_test,label="actual data")
plt.plot(X_test[sorted_indices,1],y_pred[sorted_indices],color="red",label="Gradient descent")
plt.plot(X_test[sorted_indices,1],n_predictions[sorted_indices],color="green",label="Normal equation")
plt.xlabel("Average Rooms (Standardized)")
plt.ylabel("House Value")
plt.title("Linear Regression - California Housing")

plt.legend()
plt.grid()
plt.show()
