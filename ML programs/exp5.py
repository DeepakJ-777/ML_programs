import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix


data = load_breast_cancer()

X = data.data
y = data.target.astype(int)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model1=LogisticRegression(C=np.inf,max_iter=50000)
model1.fit(X_train,y_train)
model1_pred=model1.predict(X_test)
acc_without_scaling=accuracy_score(y_test,model1_pred)

mean=np.mean(X_train,axis=0)
std=np.std(X_train,axis=0)
X_train_scaled=(X_train-mean)/std
X_test_scaled=(X_test-mean)/std

model2=LogisticRegression(C=np.inf,max_iter=50000)
model2.fit(X_train_scaled,y_train)
model2_pred=model2.predict(X_test_scaled)
acc_with_scaling=accuracy_score(y_test,model2_pred)

best_acc=0
if acc_with_scaling>acc_without_scaling:
    print("acc with scaling better",acc_with_scaling)
elif acc_without_scaling>acc_with_scaling:
    print("acc wo scaling better",acc_without_scaling)
else:
    print("both have same acc")
