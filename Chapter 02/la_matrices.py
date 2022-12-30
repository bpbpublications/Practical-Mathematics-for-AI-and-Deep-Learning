import tensorflow as tf

mat_shape = (3,3)
tf_mat_a = tf.constant([[1,3,1],[2,4,3],[4,5,1]], dtype=tf.float16, shape=mat_shape)
tf_mat_b = tf.constant([[3,1,2],[4,1,1],[2,0,1]], dtype=tf.float16, shape=mat_shape)
tf.print(type(tf_mat_a)); tf.print(tf_mat_b[0])

tf_mat_c = tf.Variable([[1,3,1],[2,4,3],[4,5,1]], dtype=tf.float16, shape=mat_shape)
tf.print(type(tf_mat_c)); tf.print(tf_mat_c[0])