import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn.metrics import silhouette_score
url = "https://raw.githubusercontent.com/gakudo-ai/open-datasets/refs/heads/main/Mall_Customers.csv"
data=pd.read_csv(url)

X=data[["Annual Income (k$)","Spending Score (1-100)"]]
scaler=StandardScaler()
X=scaler.fit_transform(X)

km=KMeans(n_clusters=5,n_init=10,random_state=42)
km_labels=km.fit_predict(X)

hc=AgglomerativeClustering(n_clusters=5,linkage="ward")
hc_labels=hc.fit_predict(X)
print("k means inertia is",km.inertia_)
print("silhouette score of k mean is",silhouette_score(X,km_labels))

print("AGG silhouette score is",silhouette_score(X,hc_labels))
plt.figure(figsize=(10,6))

plt.subplot(1,2,1)
plt.scatter(X[:,0],X[:,1],c=km_labels)
plt.xlabel("annual income")
plt.ylabel("spending score")
plt.title("kmeans")

plt.subplot(1,2,2)
plt.scatter(X[:,0],X[:,1],c=hc_labels)
plt.xlabel("annual income")
plt.ylabel("spending score")
plt.title("agglomerative")

plt.tight_layout()
plt.show()
