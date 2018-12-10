import scipy.io as sio
import sklearn.cluster as cl
from Toolbox.clusterPlot import clusterPlot as cp
import matplotlib.pyplot as plt
from Open import data
x = data.loc[:,'IFFABF':].T


y = x.index.values

clf = cl.k_means(x, n_clusters=15)
centroids = clf[0]
labels = clf[1]
print(' K-means clustering (K = 4)')
cp(x, labels, centroids, y)
plt.show()
