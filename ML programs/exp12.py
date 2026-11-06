
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = fetch_openml(name="Fashion-MNIST", version=1, as_frame=False)

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=5000, test_size=1000,random_state=42, stratify=y)

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

linear_model = SVC(kernel="linear", C=1.0)
linear_model.fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)

poly_model = SVC(kernel="poly", degree=3, C=1.0)
poly_model.fit(X_train, y_train)
poly_pred = poly_model.predict(X_test)

rbf_model = SVC(kernel="rbf", C=1.0, gamma="scale")
rbf_model.fit(X_train, y_train)
rbf_pred = rbf_model.predict(X_test)

print("LINEAR SVM")
print("Accuracy :", accuracy_score(y_test, linear_pred))
print("Precision:", precision_score(y_test, linear_pred, average="weighted"))
print("Recall   :", recall_score(y_test, linear_pred, average="weighted"))
print("F1 Score :", f1_score(y_test, linear_pred, average="weighted"))

print("\nPOLYNOMIAL SVM")
print("Accuracy :", accuracy_score(y_test, poly_pred))
print("Precision:", precision_score(y_test, poly_pred, average="weighted"))
print("Recall   :", recall_score(y_test, poly_pred, average="weighted"))
print("F1 Score :", f1_score(y_test, poly_pred, average="weighted"))

print("\nRBF SVM")
print("Accuracy :", accuracy_score(y_test, rbf_pred))
print("Precision:", precision_score(y_test, rbf_pred, average="weighted"))
print("Recall   :", recall_score(y_test, rbf_pred, average="weighted"))
print("F1 Score :", f1_score(y_test, rbf_pred, average="weighted"))
