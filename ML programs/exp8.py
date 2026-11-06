import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score
from sklearn.neighbors import KNeighborsClassifier

X,y=fetch_openml("Fashion-MNIST",version=1,return_X_y=True)
print("dataset is loaded")
X=X.astype("float32")
y=y.astype("int")
X=X/255.0
X=X[:2000]
y=y[:2000]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
accuracies=[]
k_values=[1,3,5,7]
for k in k_values:
    knn_model=KNeighborsClassifier(n_neighbors=k,n_jobs=-1)
    knn_model.fit(X_train,y_train)
    knn_pred=knn_model.predict(X_test)
    acc=accuracy_score(y_test,knn_pred)
    accuracies.append(acc)
best_index=np.argmax(accuracies)
print("best k is",k_values[best_index])
print("best acc is",accuracies[best_index])

plt.bar(k_values,accuracies)
plt.xlabel("k")
plt.ylabel("accuracy")
plt.show()
