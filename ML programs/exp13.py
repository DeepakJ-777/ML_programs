from sklearn.neural_network import MLPRegressor

import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error,r2_score

data=fetch_openml(name="wine-quality-red",version=1,as_frame=False)
X=data.data
y=data.target.astype(float)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

architectures=[(8,),(16,8),(24,16,8)]

for layers in architectures:
    model=MLPRegressor(hidden_layer_sizes=layers,max_iter=3000,random_state=42)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    print("layer is ",layers)
    print("r2 is",r2_score(y_test,y_pred))
    print("mse is ",mean_squared_error(y_test,y_pred))
