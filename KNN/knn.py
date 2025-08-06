import numpy as np
from collections import Counter
from typing import List, Union


class KNN:
    """
    K-Nearest Neighbors (KNN) classifier implementation.

    KNN is a non-parametric, lazy learning algorithm that classifies
    new data points based on the majority class among their 'k' nearest neighbors
    in the training data. The 'k' parameter determines the number of neighbors to consider.

    Parameters
    ----------
    k : int, optional
        The number of nearest neighbors to consider for classification. Defaults to 3.

    Attributes
    ----------
    k : int
        The number of nearest neighbors configured for the classifier.
    X_train : np.ndarray
        The training data features, stored after the `fit` method is called.
    y_train : np.ndarray
        The training data labels, stored after the `fit` method is called.
    """

    def __init__(self, k: int = 3):
        """
        Initializes the KNN classifier with a specified number of neighbors.
        """
        self.k = k
        self.X_train: np.ndarray
        self.y_train: np.ndarray

    def fit(
        self,
        X_train: Union[np.ndarray, List[List[float]]],
        y_train: Union[np.ndarray, List[int]],
    ):
        """
        Stores the training data for use in prediction.

        In KNN, the 'fitting' step simply involves memorizing the training dataset.

        Parameters
        ----------
        X_train : np.ndarray or list of list of float
            The training data features.
        y_train : np.ndarray or list of int
            The training data labels corresponding to `X_train`.
        """
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def predict(self, X_test: Union[np.ndarray, List[List[float]]]) -> List[int]:
        """
        Predicts the class labels for the provided test data.

        For each test data point, it calculates the Euclidean distance to all
        training data points, identifies the 'k' nearest neighbors, and assigns
        the class label based on the majority vote among these neighbors.

        Parameters
        ----------
        X_test : np.ndarray or list of list of float
            The test data features for which to make predictions. Can be a single
            sample (1D array-like) or multiple samples (2D array-like).

        Returns
        -------
        List[int]
            A list of predicted class labels for each sample in `X_test`.
        """
        X_test = np.atleast_2d(X_test)
        predictions: List[int] = []

        for test_point in X_test:
            distances = np.sqrt(np.sum((self.X_train - test_point) ** 2, axis=1))

            k_nearest_labels = self.y_train[np.argsort(distances)[: self.k]]

            prediction = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(prediction)

        return predictions
