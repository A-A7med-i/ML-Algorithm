import numpy as np
from typing import List, Tuple


class SVC:
    """
    A simple implementation of a Support Vector Classifier from scratch.

    This class provides methods to train an SVC model using gradient descent,
    make predictions, and calculate the hinge loss. The model is for binary classification
    and uses a linear kernel.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_iterations: int = 100,
        C: float = 1.0,
        verbose: bool = True,
    ):
        """
        Initializes the SVC model with hyperparameters.

        Args:
            learning_rate: The step size for the gradient descent updates.
            max_iterations: The maximum number of training iterations.
            C: The regularization parameter, which controls the trade-off between
               maximizing the margin and minimizing the classification error.
            verbose: If True, prints the loss every 10 epochs.
        """
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.weight: np.ndarray | None = None
        self.bias: float | None = None
        self.losses: List[float] = []
        self.C = C

    def _get_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the decision function scores for the given features.

        Args:
            X: The input features.

        Returns:
            An array of decision scores.
        """
        return np.dot(X, self.weight) + self.bias

    def cost_function(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculates the regularized hinge loss.

        Args:
            X: The input features.
            y: The true labels (-1 or 1).

        Returns:
            The calculated loss.
        """
        reg = 0.5 * np.dot(self.weight, self.weight)
        scores = y * self._get_scores(X)

        hinge_loss = np.maximum(0, 1 - scores)
        loss = reg + self.C * np.mean(hinge_loss)

        return loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the class labels for a given set of features.

        Args:
            X: The input features for prediction.

        Returns:
            An array of predicted class labels (0 or 1).
        """
        scores = self._get_scores(X)
        predictions = np.where(scores >= 0, 1, -1)
        return np.where(predictions <= 0, 0, 1)

    def gradient(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Calculates the gradients for the weights and bias.

        Args:
            X: The input features.
            y: The true labels (-1 or 1).

        Returns:
            A tuple containing the gradients for the weights and bias.
        """
        scores = y * self._get_scores(X)

        margin_mask = scores < 1

        dw = self.weight.copy()
        db = 0.0

        if np.any(margin_mask):
            dw -= self.C * np.mean(y[margin_mask, np.newaxis] * X[margin_mask], axis=0)
            db -= self.C * np.mean(y[margin_mask])

        return dw, db

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the SVC model using gradient descent.

        Args:
            X: The training data features.
            y: The training data labels (0 or 1).
        """
        n_features = X.shape[1]

        y = np.where(y <= 0, -1, 1)

        self.weight = np.zeros(n_features)
        self.bias = 0.0

        for i in range(self.max_iterations):
            loss = self.cost_function(X, y)
            self.losses.append(loss)

            dw, db = self.gradient(X, y)

            self.weight -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if self.verbose and (i + 1) % 10 == 0:
                print(f"Epoch: {i + 1} / {self.max_iterations}, Loss: {loss}")
