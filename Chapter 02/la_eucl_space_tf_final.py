import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# obtain metadata of Iris Dataset
iris_ds_info, iris_info = tfds.load('iris', with_info=True)
print(type(iris_ds_info))
print(iris_info.supervised_keys)
print(iris_info.features)
# Shape of features
iris_feature_shape = iris_info.features[iris_info.supervised_keys[0]].shape
print("Features Shape=", iris_feature_shape)

SAMPLE_SIZE = 6
# Capture few samples of dataset in tensor array
tf_arr = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
index = 0

# load Iris dataset
iris_ds = tfds.load('iris', split='train')
# consider first few samples from the dataset
iris_ex = iris_ds.take(SAMPLE_SIZE)
for example in iris_ex:
    feat = example["features"]
    label = example["label"]
    # Add to the array
    tf_arr = tf_arr.write(index, feat)
    index = index + 1
    vec_norm = tf.norm(feat)
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

# Compute angle between vectors using dot product
vec_angle_np = np.zeros(dtype=float, shape=(SAMPLE_SIZE, SAMPLE_SIZE))
for ref_index in range(0, SAMPLE_SIZE):
    vec_ref = tf_arr.read(ref_index)
    vec_ref_eu_nr = tf.math.reduce_euclidean_norm(vec_ref)
    for arr_index in range(0, SAMPLE_SIZE):
        vec = tf_arr.read(arr_index)
        vec_eu_nr = tf.math.reduce_euclidean_norm(vec)
        dot_prod = tf.math.reduce_sum(vec_ref * vec)
        angle_d = tf.math.acos(dot_prod/(vec_eu_nr * vec_ref_eu_nr))
        # Capture to array
        vec_angle_np[ref_index][arr_index] = angle_d.numpy()
print("Angle between vectors:\n", np.round(vec_angle_np, 2))

# calculating mean of vectors
vecs_sum = tf.constant(0, shape=iris_feature_shape, dtype=tf.float32)
for arr_index in range(0, SAMPLE_SIZE):
    vec = tf_arr.read(arr_index)
    vecs_sum = tf.math.add(vecs_sum, vec)
vecs_avg = vecs_sum / SAMPLE_SIZE
vecs_avg_np = vecs_avg.numpy()
print("Mean of sample tensors is:", np.round(vecs_avg_np, 2))

# Close the array to free the resource
tf_arr.close()
