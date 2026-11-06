
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

data = load_iris()

X = data.data[:, :2]

y=(data.target==0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
svm_model=SVC(kernel="linear",C=1.0)
svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, svm_pred))

print("\nClassification Report:")
print(classification_report(y_test, svm_pred))

w=svm_model.coef_[0]
b=svm_model.intercept_[0]

x=np.linspace(X_train[:,0].min(),X_train[:,0].max(),100)

dec_boundary= -(w[0]*x+b)/w[1]
margin1= -(w[0]*x+b-1)/w[1]
margin2= -(w[0]*x+b+1)/w[1]

plt.plot(x,dec_boundary,"k-",label="decisionBoundary")
plt.plot(x,margin1,"r--",label="margin1")
plt.plot(x,margin2,"r--",label="margin2")
plt.scatter(svm_model.support_vectors_[:,0],svm_model.support_vectors_[:,1],facecolors="none",edgecolors="black",s=120,label="support vectors")

plt.xlabel("Sepal Length (scaled)")
plt.ylabel("Sepal Width (scaled)")
plt.title("Linear SVM - Iris Dataset")
plt.legend()
plt.show()
