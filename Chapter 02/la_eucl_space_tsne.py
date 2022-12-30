import matplotlib.pyplot as plt
import numpy as np

import tensorflow_datasets as tfds

SAMPLE_SIZE = 100
MARKER_SAMPLE_SIZE = 6
IRIS_FEATURES_SHAPE = 4
iris_ds = tfds.load('iris', split='train')
iris_ex = iris_ds.take(SAMPLE_SIZE)
print(type(iris_ex))
iris_ex_np_it = tfds.as_numpy(iris_ex)
print(type(iris_ex_np_it))

# Create 2D array with shape of IRIS Features
features_np = np.zeros((0,IRIS_FEATURES_SHAPE), dtype=np.float)
labels_np = np.zeros((0), dtype=np.int)
index = 0
for iris_sample in iris_ex_np_it:
    features_np = np.insert(features_np, index, iris_sample['features'], axis=0)
    labels_np = np.insert(labels_np, index, iris_sample['label'], axis=0)
    index = index + 1

#print(features_np)
print("Shape of features:", features_np.shape)
print(labels_np)
print("Labels shape:", labels_np.shape)

from sklearn.manifold import TSNE
from matplotlib import pyplot as plot
from matplotlib import patches as mpatches

# Convert multidimensional vector to 2 dimensional vector using TSNE
X_tsne = TSNE(learning_rate=1000).fit_transform(features_np)
# Map labels to colors for plotting vectors
labels_color_dict = {0:'b', 1:'c', 2:'g'}
labels_marker_dict = {0:'^', 1:'s', 2:'P'}
print("Plotting graph\n")
ax = plt.axes()
for index in range(SAMPLE_SIZE):
    ax.scatter(x=X_tsne[index, 0], y=X_tsne[index, 1], marker=labels_marker_dict.get(labels_np[index]),
               c=labels_color_dict.get(labels_np[index]))

SETOSA_IDX = 0
setosa = ax.scatter(x=X_tsne[SETOSA_IDX, 0], y=X_tsne[SETOSA_IDX, 1],
                   marker=labels_marker_dict.get(labels_np[SETOSA_IDX]),
                   c=labels_color_dict.get(labels_np[SETOSA_IDX]))
VERSICOLOR_IDX = 2
versicolor = plt.scatter(x=X_tsne[VERSICOLOR_IDX, 0], y=X_tsne[VERSICOLOR_IDX, 1],
                       marker=labels_marker_dict.get(labels_np[VERSICOLOR_IDX]),
                       c=labels_color_dict.get(labels_np[VERSICOLOR_IDX]))
VIRGINICA_IDX = 1
virginica = plt.scatter(x=X_tsne[VIRGINICA_IDX, 0], y=X_tsne[VIRGINICA_IDX, 1],
                      marker=labels_marker_dict.get(labels_np[VIRGINICA_IDX]),
                      c=labels_color_dict.get(labels_np[VIRGINICA_IDX]))

plt.legend([setosa, versicolor, virginica], ['setosa', 'versicolor', 'virginica'])

# Plot first few points
for index in range(MARKER_SAMPLE_SIZE):
    # Mark Points
    ax.scatter(x=X_tsne[index, 0], y=X_tsne[index, 1], marker=labels_marker_dict.get(labels_np[index]),
               c='r')
    # Plot text
    vec_name = r' $\mathbf{{v_{}}}$'.format(index)
    ax.text(*np.array((X_tsne[index, 0], X_tsne[index, 1])), vec_name)

ax.set_title(label='Mapping IRIS vectors to 2D using TSNE for visualization')

plot.show()
