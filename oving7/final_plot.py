import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('agaricus-lepiota.data')
x = pd.get_dummies(data)

scaler = StandardScaler()
x = scaler.fit_transform(x)
optimal_k = 10

kmeans = MiniBatchKMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans.fit(x)
y_kmeans = kmeans.predict(x)

pca = PCA(n_components=3)
x = pca.fit_transform(x)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y_kmeans, s=50, cmap='viridis')

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title(f'K-means Clustering with {optimal_k} Clusters')

plt.show()