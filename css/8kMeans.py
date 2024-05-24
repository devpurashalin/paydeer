import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans


dataset = pd.read_csv('driver-data.csv')
x1 = dataset['mean_dist_day'].values
x2 = dataset['mean_over_speed_perc'].values
X = np.array(list(zip(x1, x2)))
print(X)

plt.plot()
plt.title('Dataset')
plt.scatter(x1, x2)
plt.show()

# code for KMeans
import matplotlib.pyplot as plt1
kmeans = KMeans(n_clusters=2)
kmeans = kmeans.fit(X)
print(kmeans.cluster_centers_)
print(kmeans.labels_)
plt1.title('K-means')
plt1.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='rainbow')
plt1.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='green')
plt1.show()

import matplotlib.pyplot as plt2
gmm = GaussianMixture(n_components=2)
gmm.fit(X)
em_prediction = gmm.predict(X)
print("EM Prediction:")
print(em_prediction)
print("EM Means:", gmm.means_)
print("\n\n")
print("EM Covariances:\n", gmm.covariances_)
# print(X)

plt2.title('Exceptation Maximum')
plt2.scatter(X[:, 0], X[:, 1], c=em_prediction, s=50)
plt2.show()
