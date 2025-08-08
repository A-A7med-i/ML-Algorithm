# Support Vector Machine (SVM) Classifier from Scratch

This directory contains a simple implementation of a linear Support Vector Machine (SVM) classifier, with the main logic in `svm.py` and an example usage in `main.py`.

---

## How SVM Works

Support Vector Machines (SVMs) are supervised learning algorithms used for binary classification. SVMs aim to find the optimal hyperplane that separates data points of different classes with the maximum margin.

The linear SVM implemented here uses the hinge loss and gradient descent for optimization.

### Mathematical Explanation

The decision function for a sample $\mathbf{x}$ is:

$$
f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b
$$

Where:

- $\mathbf{w}$ is the weight vector
- $b$ is the bias (intercept)

The predicted class is:

$$
\hat{y} = \text{sign}(f(\mathbf{x}))
$$

The **hinge loss** for a single sample is:

$$
L(\mathbf{w}, b) = \max(0, 1 - y (\mathbf{w}^T \mathbf{x} + b))
$$

The total cost function (with regularization) is:

$$
J(\mathbf{w}, b) = \frac{1}{2} \|\mathbf{w}\|^2 + C \cdot \frac{1}{m} \sum_{i=1}^m \max(0, 1 - y^{(i)} (\mathbf{w}^T \mathbf{x}^{(i)} + b))
$$

Where:

- $m$ is the number of training samples
- $C$ is the regularization parameter
- $y^{(i)} \in \{-1, 1\}$ are the true labels

Gradient descent is used to update the weights and bias to minimize this cost.

## Files

- `svm.py`: Contains the `SVC` class with `fit`, `predict`, and parameter methods.
- `main.py`: Example script showing how to use the SVM classifier.

---

## Usage

1. **Import and Initialize**

    ```python
    from SVM.svm import SVC
    model = SVC(max_iterations=100, learning_rate=0.01, C=1.0)
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
