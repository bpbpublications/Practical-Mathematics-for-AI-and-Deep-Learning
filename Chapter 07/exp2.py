import numpy as np
from numpy.linalg import pinv

np.set_printoptions(suppress=True)
cluster_samples = np.array([
    [10,15], [16,24], [25.,21], [33,28], [38,45], [40.,36]
])

outlier_smp = np.array([37.,20])
cluster_smp =
cluster_mean = np.mean(data_sample, axis=0)
print("average:", cluster_mean)

# cluster_mean vs sample_in_cluster
#cov1 = np.cov(cluster_mean, sample_in_cluster)
cov1 = np.cov(data_sample.T)
print("cov1:", cov1)
cov1_inv = pinv(cov1)
print("conv1_inv:", cov1_inv)
diff1 = cluster_mean - sample_in_cluster
mhl_dist_sqr_smp1 = np.dot(np.dot(diff1.T,cov1_inv),diff1)
print("Mahalanobis dist:", mhl_dist_sqr_smp1)
eucl_dist_sqr_smp1 = np.sum(np.square(np.subtract(cluster_mean, sample_in_cluster)))
print("Euclidean dist:", eucl_dist_sqr_smp1)

# cluster_mean vs outlier
diff2 = cluster_mean - outlier
mhl_dist_sqr_smp2 = np.dot(np.dot(diff2.T,cov1_inv),diff2)
print("Mahalanobis dist:", mhl_dist_sqr_smp2)
eucl_dist_sqr_smp2 = np.sum(np.square(np.subtract(cluster_mean, outlier)))
print("Euclidean dist:", eucl_dist_sqr_smp2)