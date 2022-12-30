import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn import datasets


def load_iris_data(num_rows=150):
    iris = datasets.load_iris()
    x = iris.data[:num_rows, :]
    y = iris.target[:num_rows]
    return x, y


def plot(x, y, column1=0, column2=1, col_name1="axis1", col_name2="axis2"):
    print("Shape before printing:", x.shape, y.shape)
    plt.scatter(x[:, column1], x[:, column2], c=y, cmap=plt.cm.Set1)
    plt.xlabel(col_name1)
    plt.ylabel(col_name2)
    plt.show()


def main():
    x, y = load_iris_data()
    tsne = TSNE(random_state=10)
    x_transformed = tsne.fit_transform(x)
    plot(x_transformed, y)


if __name__ == '__main__':
    main()
