from random import random

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import time

data = fetch_openml(name='Fashion-MNIST', version=1, as_frame=False)

X = data.data[:10000].astype('float32') / 255.0
y = data.target[:10000].astype(int)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
settings=[(0.01,64,10),(0.1,64,10),(0.1,128,10)]
for lr,bs,ep in settings:
    model=MLPClassifier(hidden_layer_sizes=(64,),learning_rate_init=lr,batch_size=bs,max_iter=ep,random_state=42)
    start = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    tot_time= time.time()-start
    accuracy = accuracy_score(y_test, y_pred)

    print("\nLearning rate:", lr)
    print("Batch size:", bs)
    print("Epochs:", model.n_iter_)
    print("Accuracy:",accuracy)
    print("Training time:",tot_time)
