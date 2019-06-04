import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from itertools import cycle

with open('data/plots_some', 'r') as f:
  plots_array = f.read().split('<EOS>')

with open('data/titles', 'r') as f:
    titles_array = f.readlines()

vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(plots_array)
feature_names = vectorizer.get_feature_names()

print(x.shape)

with open('vocab', 'w') as f:
  f.write(', '.join(feature_names))


# print(X)
X2 = x.toarray()
# X2 = np.array(list(map(lambda item: item[0:4200], X)))

with open('debug', 'w') as f:
  f.write(', '.join(str(v) for v in X2[1]))


# centers = [[1, 1], [-1, -1], [1, -1]]
# X2, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)

# print(X2)
# print('vector len is ', len(feature_names)) 

# bandwidth = estimate_bandwidth(X,)

bandwidth = estimate_bandwidth(X2)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X2)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)


with open('n_clusters_', 'w') as f:
  f.write("number of estimated clusters: %d" % n_clusters_)

# print("number of estimated clusters : %d" % n_clusters_)

# #############################################################################
# Plot result

plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X2[my_members, 0], X2[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=5)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()