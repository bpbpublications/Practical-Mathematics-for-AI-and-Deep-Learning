import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
iris_ds = tfds.load('iris', split='train')  # load Iris dataset
SAMPLE_SIZE = 6  # consider few samples from the dataset
iris_ex = iris_ds.take(SAMPLE_SIZE)  # consider first few samples from the dataset
# Tensor array to store dataset samples
tf_arr = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
index = 0
for sample in iris_ex:
    feat = sample["features"]
    label = sample["label"]
    tf_arr = tf_arr.write(index, feat)  # Add sample to array
    index = index + 1
    vec_norm = tf.norm(feat)  # Calculate norm of the vector
    tf.print("label=", label, ' vector v', index, '=', feat, " Euclidean Norm=", vec_norm, sep="")

# Distance between vectors
vec_dis_np = np.zeros(dtype=float, shape=(SAMPLE_SIZE, SAMPLE_SIZE))
# Distance between vectors
for ref_index in range(0, SAMPLE_SIZE):
    vec_ref = tf_arr.read(ref_index)
    for arr_index in range(0, SAMPLE_SIZE):
        vec = tf_arr.read(arr_index)
        vec_sub = tf.math.subtract(vec_ref, vec)
        distance = tf.math.reduce_euclidean_norm(vec_sub)
        # Capture to array
        vec_dis_np[ref_index][arr_index] = distance.numpy()
# Print distance between vectors
print("Distance between vectors:\n", np.round(vec_dis_np, 2))