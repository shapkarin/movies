import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from itertools import cycle
from joblib import dump, load

with open('data/plots_some', 'r') as f:
  plots_array = f.read().split('<EOS>')

with open('data/titles', 'r') as f:
    titles_array = f.readlines()

vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(plots_array)
feature_names = vectorizer.get_feature_names()

with open('vocab', 'w') as f:
  f.write(', '.join(feature_names))


# print(X)
# X2 = x.toarray()
X2 = np.array(list(map(lambda item: item[0:1000], x.toarray())))

with open('results/debug.txt', 'w') as f:
  f.write(', '.join(str(v) for v in X2[1]))


# centers = [[1, 1], [-1, -1], [1, -1]]
# X2, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)

# print(X2)
# print('vector len is ', len(feature_names)) 

# bandwidth = estimate_bandwidth(X,)

bandwidth = estimate_bandwidth(X2)

# ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms = MeanShift(bandwidth=bandwidth)
ms.fit(X2)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print('labels', labels)
print('cluster_centers', cluster_centers)


with open('results/n_clusters_.txt', 'w') as f:
  f.write("number of estimated clusters: %d" % n_clusters_)


dump(X2, 'results/X2.joblib')
dump(ms, 'results/clusters.joblib')
