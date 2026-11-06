

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = fetch_openml(name="adult", version=2, as_frame=False)

X = data.data
y = data.target

valid = np.all(X != "?", axis=1)
X = X[valid]
y = y[valid]

num_cols=[0,2,4,10,11,12]
cat_cols=[1,3,5,6,7,8,9,13]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
encoder=OneHotEncoder(handle_unknown="ignore",sparse_output=False)
X_train_cat=encoder.fit_transform(X_train[:,cat_cols])
X_test_cat=encoder.transform(X_test[:,cat_cols])

X_train_num=X_train[:,num_cols].astype(float)
X_test_num=X_test[:,num_cols].astype(float)
X_train=np.concatenate((X_train_cat,X_train_num),axis=1)
X_test=np.concatenate((X_test_cat,X_test_num),axis=1)
#Logisitic reg aane always standardscalar then proceed
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
lr_model=LogisticRegression(max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)

# 8. Decision Tree
dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)

dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

print("LOGISTIC REGRESSION")
print("Accuracy :", accuracy_score(y_test, lr_pred))
print("Precision:", precision_score(y_test, lr_pred, average="weighted"))
print("Recall   :", recall_score(y_test, lr_pred, average="weighted"))
print("F1 Score :", f1_score(y_test, lr_pred, average="weighted"))

print("\nDECISION TREE")
print("Accuracy :", accuracy_score(y_test, dt_pred))
print("Precision:", precision_score(y_test, dt_pred, average="weighted"))
print("Recall   :", recall_score(y_test, dt_pred, average="weighted"))
print("F1 Score :", f1_score(y_test, dt_pred, average="weighted"))

plt.figure(figsize=(10,6))

plot_tree(dt_model,max_depth=3,filled=True,rounded=True)
plt.show()
