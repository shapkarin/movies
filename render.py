import numpy as np
import matplotlib.pyplot as plt
from joblib import load
from itertools import cycle

X2 = load('results/X2.joblib') 
ms = load('results/clusters.joblib') 

labels = ms.labels_
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
cluster_centers = ms.cluster_centers_

print("number of estimated clusters : %d" % n_clusters_)

#############################################################################
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
