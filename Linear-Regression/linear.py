from typing import List, Optional
import numpy as np


class LinearRegression:
    """
    Implements Linear Regression using gradient descent.

    This model learns a linear relationship between input features and a target variable.
    It minimizes the Mean Squared Error (MSE) by iteratively adjusting weights and bias.

    Parameters
    ----------
    max_iterations : int
        The maximum number of training steps.
    learning_rate : float
        The step size for updating weights and bias during training.
    verbose : bool, optional
        If True, training progress (MSE per epoch) will be printed.
        Defaults to True.

    Attributes
    ----------
    weights : np.ndarray or None
        The learned coefficients for each feature after training. None until `fit` is called.
    bias : float or None
        The learned intercept term after training. None until `fit` is called.
    history : list of float
        A record of the cost (MSE) at each training iteration.
    """

    def __init__(
        self,
        max_iterations: int,
        learning_rate: float,
        verbose: bool = True,
    ):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[float] = None
        self.history: List[float] = []

    def cost_function(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculates the cost (Mean Squared Error) for the current predictions.

        Parameters
        ----------
        y_pred : np.ndarray
            The model's predicted values.
        y_true : np.ndarray
            The actual true values.

        Returns
        -------
        float
            The calculated cost (MSE).
        """
        m = len(y_true)
        return (1 / (2 * m)) * np.sum((y_pred - y_true) ** 2)

    @staticmethod
    def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculates the Mean Squared Error (MSE) between predicted and true values.

        Parameters
        ----------
        y_pred : np.ndarray
            The model's predicted values.
        y_true : np.ndarray
            The actual true values.

        Returns
        -------
        float
            The Mean Squared Error.
        """
        return np.mean((y_true - y_pred) ** 2)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """
        Trains the Linear Regression model using gradient descent.

        Initializes weights to zeros and bias to zero, then iteratively
        updates them to minimize the cost function.

        Parameters
        ----------
        X : np.ndarray
            The training features (input data).
        y : np.ndarray
            The training targets (output data).

        Returns
        -------
        self : LinearRegression
            The trained LinearRegression instance, allowing for method chaining.
        """
        self.m, self.n = X.shape

        self.weights = np.zeros(self.n)
        self.bias = 0

        if self.verbose:
            print("-------------------Training Start ------------------")

        for i in range(self.max_iterations):
            y_pred = X.dot(self.weights) + self.bias

            dw = (1 / self.m) * np.dot(X.T, (y_pred - y))
            db = (1 / self.m) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            cost = self.cost_function(y_pred, y)
            self.history.append(cost)

            mse = self.mse(y, y_pred)

            if self.verbose:
                print(f"Epoch {i + 1}/{self.max_iterations}, MSE: {mse:.4f}")
        return self

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """
        Generates predictions for new data using the trained model.

        Parameters
        ----------
        x_test : np.ndarray
            The input features for which to make predictions.

        Returns
        -------
        np.ndarray
            The predicted target values.

        Raises
        ------
        ValueError
            If the model has not been trained yet (i.e., `fit()` has not been called).
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")

        return np.dot(x_test, self.weights) + self.bias

    def get_params(self) -> dict:
        """
        Retrieves the current parameters of the model.

        Returns
        -------
        dict
            A dictionary containing the learned weights, bias, learning rate,
            and maximum iterations.
        """
        return {
            "Weights": self.weights,
            "Bias": self.bias,
            "Learning Rate": self.learning_rate,
            "Max Iterations": self.max_iterations,
        }
