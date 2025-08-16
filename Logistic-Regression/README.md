# Logistic Regression from Scratch

This directory contains a simple implementation of Logistic Regression using gradient descent, with the main logic in `logistic.py` and an example usage in `main.py`.

---

## How Logistic Regression Works

Logistic Regression is a supervised learning algorithm used for binary classification. It models the probability that a given input belongs to the positive class using the logistic (sigmoid) function.

The model outputs probabilities between 0 and 1, which are then thresholded (commonly at 0.5) to assign class labels.

### Mathematical Explanation

The hypothesis (predicted probability) for a single sample is:

$$
\hat{y} = \sigma(\mathbf{w}^T \mathbf{x} + b)
$$

Where:

- $\mathbf{x}$ is the feature vector
- $\mathbf{w}$ is the weight vector (coefficients)
- $b$ is the bias (intercept)
- $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the sigmoid function

The model is trained by minimizing the **binary cross-entropy loss** (log loss):

$$
J(\mathbf{w}, b) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]
$$

Where:

- $m$ is the number of training samples
- $\hat{y}^{(i)}$ is the predicted probability for sample $i$
- $y^{(i)}$ is the true label for sample $i$

**Gradient Descent** is used to iteratively update the weights and bias to minimize the loss:

$$
w_j := w_j - \alpha \frac{\partial J}{\partial w_j} \\
b := b - \alpha \frac{\partial J}{\partial b}
$$

Where $\alpha$ is the learning rate.

## Files

- `logistic.py`: Contains the `LogisticRegression` class with `fit`, `predict`, and parameter methods.
- `main.py`: Example script showing how to use the Logistic Regression model.

---

## Usage

1. **Import and Initialize**

    ```python
    from logistic import LogisticRegression
    model = LogisticRegression(max_iterations=100, learning_rate=0.1)
    ```

2. **Fit the Model**

    ```python
    model.fit(X_train, y_train)
    ```

3. **Make Predictions**

    ```python
    predictions = model.predict(X_test)
    ```

Or simply run the example:

```bash
python main.py
```

---

## Requirements

- Python 3.x
- NumPy
- scikit-learn (for generating example data and metrics)

Install dependencies with:

```bash
pip install numpy scikit-learn
```

---
