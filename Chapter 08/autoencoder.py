import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomUniform


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


class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(
                units=2, activation='relu',
                kernel_initializer=RandomUniform(minval=0., maxval=1.,
                                                 seed=10)),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(
                units=4, activation='relu',
                kernel_initializer=RandomUniform(minval=0., maxval=1.,
                                                 seed=10)),
        ])
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def main():
    x, y = load_iris_data()
    # Split the data into train & test
    x_train, x_test, y_train, y_test =\
        train_test_split(x, y, random_state=10, test_size=.3)
    print("x_train, x_test,", x_train.shape, x_test.shape)

    # Training the model
    autoencod_model = Autoencoder()
    autoencod_model.compile(optimizer='sgd', loss=losses.MeanSquaredError())
    autoencod_model.fit(
        x_train, x_train, epochs=40, batch_size=30,
        validation_data=(x_test, x_test)
    )
    autoencod_model.summary()
    # Obtain encoded information of test set
    encoded_vect = autoencod_model.encoder(x).numpy()
    decoded_vect = autoencod_model.decoder(encoded_vect).numpy()

    # Plotting points
    plot(encoded_vect, y)


if __name__ == '__main__':
    main()
