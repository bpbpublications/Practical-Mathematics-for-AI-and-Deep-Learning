from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

NUM_ROWS = 150


def load_iris_data(num_rows=150):
    iris = datasets.load_iris()
    x = iris.data[:num_rows, :]
    y = iris.target[:num_rows]
    return x, y


def plot(x, y, column1=0, column2=1, col_name1="axis1", col_name2="axis2"):
    plt.scatter(x[:, column1], x[:, column2], c=y, cmap=plt.cm.Set1)
    plt.xlabel(col_name1)
    plt.ylabel(col_name2)
    plt.show()


def main():
    x, y = load_iris_data()
    # Covariance matrix
    cov_mat = np.cov(x.T)
    print("cov_mat:", cov_mat)

    # Obtain eigen values/vectors
    eigen_val, eigen_vect = np.linalg.eig(cov_mat)
    print("eigen_val:\n", eigen_val)
    print("eigen_vect:\n", eigen_vect)

    # Eigen Decomposition of covariance matrix
    mat_p = eigen_vect
    mat_p_inv = np.linalg.inv(mat_p)
    mat_d = np.array([
        [eigen_val[0],0,0,0],
        [0,eigen_val[1],0,0],
        [0,0,eigen_val[2],0],
        [0,0,0,eigen_val[3]]
    ])
    mat_result = np.matmul(np.matmul(mat_p,mat_d),mat_p_inv)
    print("Diagonal Mat\n", mat_d)
    print("Resultant mat\n", mat_result)

    # Select two principal components
    eigen_vect0 = eigen_vect[:, 0]
    eigen_vect1 = eigen_vect[:, 1]
    print("Eigen vector0:\n", eigen_vect0)
    print("Eigen vector1:\n", eigen_vect1)

    trans_mat = np.array([eigen_vect0, eigen_vect1]).T
    print("Linear Transformation: \n", trans_mat)

    # Verification of eigen properties
    mat1 = np.matmul(cov_mat, eigen_vect0) / eigen_val[0]
    mat2 = np.matmul(cov_mat, eigen_vect1) / eigen_val[1]
    print("Matrix1:\n", mat1)
    print("Matrix2:\n", mat2)

    # Check orthogonality of eigen vectors
    dot_prod1 = np.dot(eigen_vect0, eigen_vect1)
    print("dot_prod1:", np.array(dot_prod1))

    # Checking the norm
    norm_vect0 = np.linalg.norm(eigen_vect0)
    norm_vect1 = np.linalg.norm(eigen_vect1)
    print("norm_vect0:", norm_vect0)
    print("norm_vect1:", norm_vect1)

    # Transform the data points
    x_reduced = np.matmul(x, trans_mat)
    #plot(x_reduced, y)

    # Plotting of original data
    #plot(x, y, column1=0, column2=1, col_name1="sepal_length", col_name2="sepal_width")
    #plot(x, y, column1=2, column2=3, col_name1="petal_length", col_name2="petal_width")

    # Trace
    cov_mat_tr = np.trace(cov_mat)
    eigen_val_sum = np.sum(eigen_val)
    print("Covariance Matrix Trace:", cov_mat_tr)
    print("Eigen value sum:", eigen_val_sum)
    print("variance e0:", eigen_val[0] * 100./cov_mat_tr)
    eigen_sum_e01 = eigen_val[0] + eigen_val[1]
    print("variance e01:", eigen_sum_e01 * 100. / cov_mat_tr)
    eigen_sum_e012 = eigen_val[0] + eigen_val[1] + eigen_val[2]
    print("variance e012:", eigen_sum_e012 * 100. / cov_mat_tr)



if __name__ == '__main__':
    main()
