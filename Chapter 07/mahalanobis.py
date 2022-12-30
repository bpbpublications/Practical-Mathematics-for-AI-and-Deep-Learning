import numpy as np
from numpy.linalg import pinv


def mahalanobis_dist_sqr(sample1, sample2, cov_matrix_inv):
    mean_smp_diff = sample1 - sample2
    return np.dot(np.dot(mean_smp_diff.T,cov_matrix_inv),mean_smp_diff)


def euclidean_dist_sqr(sample1, sample2):
    return np.sum(np.square(np.subtract(sample1, sample2)))


np.set_printoptions(precision=3, suppress=True)
cluster_samples = np.array([
    [10,15], [16,24], [25.,21], [33,28], [38,45], [40.,36], [37.,20]
])
cluster_mean = np.mean(cluster_samples, axis=0)
print("Cluster mean:", cluster_mean)

# Calculating covariance matrix
clust_cov = np.cov(cluster_samples.T)
clust_cov_inv = pinv(clust_cov)
print("cluster covariance matrix: \n", clust_cov)
print("cluster covariance matrix inverse: \n", clust_cov_inv)

outlier_smp = cluster_samples[6]
cluster_smp = cluster_samples[5]
# cluster mean vs cluster sample
mhl_dist_sqr_smp1 = mahalanobis_dist_sqr(cluster_mean, cluster_smp, clust_cov_inv)
print("Mahalanobis Distance: \n", mhl_dist_sqr_smp1)
eucl_dist_sqr_smp1 = euclidean_dist_sqr(cluster_mean, cluster_smp)
print("Euclidean Distance: \n", eucl_dist_sqr_smp1)

# cluster mean vs cluster outlier
mhl_dist_sqr_smp2 = mahalanobis_dist_sqr(cluster_mean, outlier_smp, clust_cov_inv)
print("Mahalanobis Distance Square: \n", mhl_dist_sqr_smp2)
eucl_dist_sqr_smp2 = euclidean_dist_sqr(cluster_mean, outlier_smp)
print("Euclidean Distance Square: \n", eucl_dist_sqr_smp2)

# Plotting of samples
import matplotlib.pyplot as plt
cluster_samples = np.array([
    [10,15], [16,24], [25.,21], [33,28], [38,45]
])
plt.scatter(x=cluster_samples[:,0], y=cluster_samples[:,1], c='gray', s=40)
plt.scatter(x=cluster_mean[0], y=cluster_mean[1], marker="v", c='black', s=40)
plt.scatter(x=outlier_smp[0], y=outlier_smp[1], marker="s", c='dimgray', s=40)
plt.scatter(x=cluster_smp[0], y=cluster_smp[1], marker="*", c='dimgray', s=40)
plt.show()
