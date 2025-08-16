from typing import List, Tuple
import numpy as np


class LogisticRegression:
    """
    A simple implementation of Logistic Regression from scratch.

    This class provides methods to train a logistic regression model using gradient descent,
    make predictions, and calculate performance metrics.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_iterations: int = 100,
        epsilon: float = 1e-10,
        verbose: bool = True,
    ):
        """
        Initializes the LogisticRegression model with hyperparameters.

        Args:
            learning_rate: The step size for the gradient descent updates.
            max_iterations: The maximum number of training iterations.
            epsilon: A small value to prevent division by zero in the cost function.
            verbose: If True, prints the loss every 10 epochs.
        """
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.verbose = verbose
        self.losses: List[float] = []
        self.weight: np.ndarray | None = None
        self.bias: float | None = None

    def cost_function(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates the binary cross-entropy loss.

        Args:
            y_true: The true labels.
            y_pred: The predicted probabilities.

        Returns:
            The calculated cost.
        """
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        cost = -np.mean(
            y_true * np.log(y_pred + self.epsilon)
            + (1 - y_true) * np.log(1 - y_pred + self.epsilon)
        )

        return cost

    def gradient(
        self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Calculates the gradients for the weights and bias.

        Args:
            X: The input features.
            y_true: The true labels.
            y_pred: The predicted probabilities.

        Returns:
            A tuple containing the gradients for the weights and bias.
        """
        m = X.shape[0]
        dw = (1 / m) * np.dot(X.T, (y_pred - y_true))
        db = (1 / m) * np.sum(y_pred - y_true)

        return dw, db

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Calculates the sigmoid activation function.

        Args:
            z: The linear combination of weights and features.

        Returns:
            The output of the sigmoid function.
        """
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the logistic regression model using gradient descent.

        Args:
            X: The training data features.
            y: The training data labels.
        """
        n_sample = X.shape[0]
        n_features = X.shape[1]

        self.weight = np.zeros(n_features)
        self.bias = 0

        for i in range(self.max_iterations):
            y_pred = np.dot(X, self.weight) + self.bias
            y_pred = self.sigmoid(y_pred)

            cost = self.cost_function(y, y_pred)
            self.losses.append(cost)

            dw, db = self.gradient(X, y, y_pred)

            self.weight -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if self.verbose and (i + 1) % 10 == 0:
                print(f"Epoch: {i + 1} / {self.max_iterations}, Loss: {cost}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the probability of a sample belonging to the positive class.

        Args:
            X: The input features for prediction.

        Returns:
            An array of predicted probabilities.
        """
        linear_pred = np.dot(X, self.weight) + self.bias
        return self.sigmoid(linear_pred)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predicts the class labels for a given set of features.

        Args:
            X: The input features for prediction.
            threshold: The threshold to classify a sample as positive.

        Returns:
            An array of predicted class labels (0 or 1).
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
