import tensorflow_datasets as tfds
import numpy as np

IRIS_FEATURES_CNT = 4


def get_features_labels(iris_dataset):
    iris_feat = np.zeros((0, IRIS_FEATURES_CNT), dtype=np.float)
    iris_label = np.zeros((0), dtype=np.int)
    # Capture training data in array
    index = 0
    for iris_samp in iris_dataset.as_numpy_iterator():
        iris_feat = np.insert(iris_feat, index, iris_samp['features'], axis=0)
        iris_label = np.insert(iris_label, index, iris_samp['label'], axis=0)
        index = index + 1
    return iris_feat, iris_label


def main():
    # Load Iris dataset of 150 samples into train and test test
    iris_train = tfds.load('iris', split='train[:120]')
    iris_test = tfds.load('iris', split='train[120:]')
    # Extract features and labels of iris samples
    iris_train_feat, iris_train_label = get_features_labels(iris_train)
    iris_test_feat, iris_test_label = get_features_labels(iris_test)
    # Assign labels to test samples based on k-nearest neighbours
    for k_val in range(3, 34, 3):
        test_knn_labels = np.zeros((0), dtype=np.int)
        outer_index = 0
        # Calculate Euclidean distance between a test vector & all training vectors
        # Select k-nearest neighbours based on the distance
        for test_feat_samp in iris_test_feat:
            eu_dis = np.zeros((0), dtype=np.float)
            inner_index = 0
            for train_feat_samp in iris_train_feat:
                euclidean_distance = np.linalg.norm(train_feat_samp - test_feat_samp)
                eu_dis = np.insert(eu_dis, inner_index, euclidean_distance, axis=0)
                inner_index = inner_index + 1  # Move to next train vector
            sorted_index = np.argsort(eu_dis)  # Sort based on euclidean distance
            # labels of k-shortest distance
            nearest_k_labels = iris_train_label[sorted_index[0:k_val]]
            (labels, count) = np.unique(nearest_k_labels, return_counts=True)
            # Assign maximum repeated label
            test_samp_knn_pred_label = labels[np.argmax(count)]
            test_knn_labels = np.insert(
                test_knn_labels, outer_index, test_samp_knn_pred_label, axis=0)
            outer_index = outer_index + 1  # Move to next test vector
        correct_prediction = np.sum(test_knn_labels == iris_test_label)
        wrong_prediction = iris_test_label.shape[0] - correct_prediction

        print("k-val:", k_val, "\nPD:", test_knn_labels, "\nGT:", iris_test_label)
        print("correct prediction", correct_prediction, ",wrong prediction", wrong_prediction)


if __name__ == '__main__':
    main()



