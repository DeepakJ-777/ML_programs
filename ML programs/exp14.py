import time
import matplotlib.pyplot as plt
from numpy.ma.core import tanh
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

data=fetch_openml(name="mnist_784",version=1,as_frame=False)
X=data.data[:1000].astype("float32")/255.0
y=data.target[:1000].astype(int)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
activations=['logistic','relu','tanh']
acc=[]

for act in activations:
    model=MLPClassifier(hidden_layer_sizes=(64,),activation=act,random_state=42,max_iter=20)
    start=time.time()
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    tot_time=time.time()-start
    print("activation function here is",act)
    print("acc is",accuracy_score(y_test,pred))
    print("time taken is",tot_time)
    acc.append(accuracy_score(y_test,pred))
plt.bar(activations,acc)
plt.xlabel("activation functions")
plt.ylabel("accuracies")
plt.show()
