from tensorflow.keras.datasets import mnist
import tensorflow as tf
from tensorflow.keras import layers, Model
import time
import numpy as np
from matplotlib.pylab import plt

def vae_encoder(latent_dim):
    epsilon = layers.Input(shape=latent_dim)
    img = layers.Input(shape=(28, 28, 1)) #For MNIST
    
    x = layers.Conv2D(filters=32, kernel_size=3, 
                      strides=2, padding='same')(img)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(filters=64, kernel_size=3
                      , strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)    
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
    return Model(inputs = [img, epsilon], outputs = [z_mean, z_log_var, z])
    
def vae_decoder(latent_dim):
    z = layers.Input(shape=(latent_dim,))
    x = layers.Dense(7 * 7 * 64, activation="relu")(z)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(filters=64, kernel_size=3,
                               strides=2, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2DTranspose(filters=32, kernel_size=3,
                               strides=2, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, 
                activation="sigmoid",padding="same")(x)
    return Model(z, decoder_outputs)

class VAEModel:
    def __init__(self, encoder, decoder, noise_dim):
        self.encoder = encoder
        self.decoder = decoder
        self.noise_dim = noise_dim
        self.optimizer = tf.keras.optimizers.Adam()
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        
    def __train_step__(self, data, batch_size):
        noise = tf.random.normal([batch_size, self.noise_dim])

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder([data, noise])
            reconstruction = self.decoder(z)

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                     tf.keras.losses.binary_crossentropy(data, reconstruction), 
                     axis=(1, 2)
                )
            )
            '''
            #OR as sumsquared
            reconstruction_loss = tf.reduce_mean(
                tf.square(tf.norm(data-reconstruction))
            )
            '''
            kl_loss = tf.reduce_sum(
            -0.5 * (z_log_var  - tf.exp(z_log_var) - tf.square(z_mean) + 1),
                  axis=1)
            kl_loss = tf.reduce_mean(kl_loss)
            total_loss = reconstruction_loss + kl_loss

        weights = self.encoder.trainable_variables + self.decoder.trainable_variables
        grads = tape.gradient(total_loss, weights)
        self.optimizer.apply_gradients(zip(grads, weights))
        self.total_loss_tracker.update_state(total_loss)
    

    def train(self, dataset, epochs, batch_size):
        for epoch in range(epochs):
            start = time.time()
            print('starting epoch {}'.format(epoch))
            
            for image_batch in dataset:   
                if image_batch.shape[0]==batch_size:
                    self.__train_step__(image_batch, batch_size)
            print ('Time for epoch {} is {} sec, loss {}'.format(epoch + 1, time.time()-start, self.total_loss_tracker.result()))

    def plot_latent_space(self, n=30, figsize=15):
        # display a n*n 2D manifold of digits
        digit_size = 28
        scale = 1.0
        figure = np.zeros((digit_size * n, digit_size * n))
        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        grid_x = np.linspace(-scale, scale, n)
        grid_y = np.linspace(-scale, scale, n)[::-1]

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi]])
                x_decoded = self.decoder.predict(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[
                    i * digit_size : (i + 1) * digit_size,
                    j * digit_size : (j + 1) * digit_size,
                ] = digit

        plt.figure(figsize=(figsize, figsize))
        start_range = digit_size // 2
        end_range = n * digit_size + start_range
        pixel_range = np.arange(start_range, end_range, digit_size)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.imshow(figure, cmap="Greys_r")
        plt.show()



