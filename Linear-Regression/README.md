# Linear Regression from Scratch

This directory contains a simple implementation of Linear Regression using gradient descent, with the main logic in `linear.py` and an example usage in `main.py`.

---

## How Linear Regression Works

Linear Regression is a supervised learning algorithm used to model the relationship between a dependent variable (target) and one or more independent variables (features) by fitting a linear equation to observed data.

The goal is to find the best-fitting straight line (in higher dimensions, a hyperplane) that predicts the target variable as accurately as possible.

### Mathematical Explanation

The hypothesis (prediction) for a single sample is:

$$
\hat{y} = w_1 x_1 + w_2 x_2 + ... + w_n x_n + b = \mathbf{w}^T \mathbf{x} + b
$$

Where:

- $\mathbf{x}$ is the feature vector
- $\mathbf{w}$ is the weight vector (coefficients)
- $b$ is the bias (intercept)

The model is trained by minimizing the **Mean Squared Error (MSE)** cost function:

$$
J(\mathbf{w}, b) = \frac{1}{2m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})^2
$$

Where:

- $m$ is the number of training samples
- $\hat{y}^{(i)}$ is the predicted value for sample $i$
- $y^{(i)}$ is the true value for sample $i$

**Gradient Descent** is used to iteratively update the weights and bias to minimize the cost:

$$
w_j := w_j - \alpha \frac{\partial J}{\partial w_j} \\
b := b - \alpha \frac{\partial J}{\partial b}
$$

Where $\alpha$ is the learning rate.

---

## Files

- `linear.py`: Contains the `LinearRegression` class with `fit`, `predict`, and parameter methods.
- `main.py`: Example script showing how to use the Linear Regression model.

---

## Usage

1. **Import and Initialize**

    ```python
    from linear import LinearRegression
    model = LinearRegression(max_iterations=100, learning_rate=0.01)
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
- scikit-learn (for generating example data)

Install dependencies with:

```bash
pip install numpy scikit-learn
```

---
