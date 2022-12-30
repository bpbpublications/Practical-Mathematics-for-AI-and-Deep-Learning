import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

IRIS_FEATURES_SHAPE = 4

# load Iris dataset
iris_train = tfds.load('iris', split='train[:120]')
iris_test = tfds.load('iris', split='train[120:]')

train_feat = np.zeros((0,IRIS_FEATURES_SHAPE), dtype=np.float)
train_label = np.zeros((0), dtype=np.int)
test_feat = np.zeros((0,IRIS_FEATURES_SHAPE), dtype=np.float)
test_label = np.zeros((0), dtype=np.int)

# Capture training data in array
index = 0
for train_samp in iris_train.as_numpy_iterator():
    train_feat = np.insert(train_feat, index, train_samp['features'], axis=0)
    train_label = np.insert(train_label, index, train_samp['label'], axis=0)
    index = index + 1
# Capture test data in array
index = 0
for test_samp in iris_test.as_numpy_iterator():
    test_feat = np.insert(test_feat, index, test_samp['features'], axis=0)
    test_label = np.insert(test_label, index, test_samp['label'], axis=0)
    index = index + 1

for k_val in range(3,34,3):
    test_knn_labels = np.zeros((0), dtype=np.int)
    outer_index = 0
    ''' Calculate Euclidean distance between a test vector & all training vectors'''
    for test_feat_samp in test_feat:
        eu_dis = np.zeros((0), dtype=np.float)
        inner_index = 0
        ''' Calculate Euclidean distance between a test vector & all training vectors '''
        for train_feat_samp in train_feat:
            euclidean_distance = np.linalg.norm(train_feat_samp - test_feat_samp)
            eu_dis = np.insert(eu_dis, inner_index, euclidean_distance, axis=0)
            inner_index = inner_index + 1
        ''' Obtain index of euclidean distance elements in non-decreasing order '''
        sorted_index = np.argsort(eu_dis)
        ''' Obtain labels of first K shortest distance vectors '''
        nearest_k_labels = train_label[sorted_index[0:k_val]]
        ''' From K vectors, count vectors for each of the label '''
        (labels, count) = np.unique(nearest_k_labels, return_counts=True)
        ''' Assign maximum repeated label for the test vector '''
        test_samp_knn_pred_label = labels[np.argmax(count)]
        test_knn_labels = np.insert(test_knn_labels, outer_index, test_samp_knn_pred_label, axis=0)
        outer_index = outer_index + 1

    correct_prediction = np.sum(test_knn_labels == test_label)
    wrong_prediction = test_label.shape[0] - correct_prediction

    print("k-val:", k_val, "\nPD:", test_knn_labels, "\nGT:", test_label)
    print("correct prediction", correct_prediction, ",wrong prediction", wrong_prediction)
