import tensorflow as tf
# Instantiating built in vgg16 model
vgg16 = tf.keras.applications.vgg16.VGG16(
    include_top=True, weights=None,
    classes=1000, classifier_activation='softmax'
)
print(vgg16.summary())
