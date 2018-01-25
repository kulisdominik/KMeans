from sklearn import datasets

iris = datasets.load_iris()
x = iris.data[:, :4]

digits = iris.target


import matplotlib.pyplot as plt


from sklearn.cluster import KMeans
wcss = []

kmeans = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 500, n_init = 10)
y_kmeans = kmeans.fit_predict(x)


plt.scatter(x[y_kmeans == 0, 2], x[y_kmeans == 0, 3], s = 100, c = 'red', label = 'czerwony')
plt.scatter(x[y_kmeans == 1, 2], x[y_kmeans == 1, 3], s = 100, c = 'blue', label = 'niebieski')
plt.scatter(x[y_kmeans == 2, 2], x[y_kmeans == 2, 3], s = 100, c = 'green', label = 'zielony')
plt.scatter(x[y_kmeans == 3, 2], x[y_kmeans == 3, 3], s = 100, c = 'purple', label = 'fiolotowy')

from sklearn import metrics

print(metrics.homogeneity_score(digits, y_kmeans))
print(metrics.adjusted_rand_score(digits, y_kmeans))

plt.scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:,3], s = 100, c = 'yellow', label = 'Centroids')
plt.legend()
plt.show()


