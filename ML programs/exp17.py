
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

digits = load_digits()
X = digits.data

scaler = StandardScaler()
X = scaler.fit_transform(X)
inertia=[]
silhouette=[]
ks=range(2,16)
for k in ks:
    model=KMeans(n_clusters=k,n_init=10,random_state=42)
    km_labels=model.fit_predict(X)
    inertia.append(model.inertia_)
    silhouette.append(silhouette_score(X,km_labels))
    print("inertia is",model.inertia_)
    print("silhouette score is",silhouette_score(X,km_labels))

plt.figure(figsize=(10,6))
plt.plot(ks,inertia,marker="o")
plt.xlabel("k values")
plt.ylabel("inertias")
plt.title("elbow method")
plt.grid()
plt.show()

plt.figure(figsize=(10,6))
plt.plot(ks,silhouette,marker="o")
plt.xlabel("k values")
plt.ylabel("sils")
plt.title("sil method")
plt.grid()
plt.show()
