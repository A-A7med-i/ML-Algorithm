# Machine Learning Algorithms from Scratch

This repository contains implementations of fundamental machine learning algorithms written from scratch in Python. Each algorithm is organized in its own directory with clear code and example usage.

---

## Project Structure

```
ML-Algorithm/
├── KNN/
│   ├── knn.py          # Implementation of K-Nearest Neighbors
│   ├── main.py         # Example usage of KNN
│   └── README.md       # Documentation for KNN
├── Linear-Regression/
│   ├── linear.py       # Implementation of Linear Regression
│   ├── main.py         # Example usage of Linear Regression
│   └── README.md       # Documentation for Linear Regression
├── Logistic-Regression/
│   ├── logistic.py     # Implementation of Logistic Regression
│   ├── main.py         # Example usage of Logistic Regression
│   └── README.md       # Documentation for Logistic Regression
├── SVM/
│   ├── svm.py          # Implementation of Support Vector Machine
│   ├── main.py         # Example usage of SVM
│   └── REAMD.md        # Documentation for SVM
│
├── requirements.txt     # Project dependencies
├── setup.py             # Package setup
├── LICENSE              # LICENSE File
├── .gitignore           # Git ignore rules
└── README.Md           # Top-level documentation for the repository
```

---

## Implemented Algorithms

### 1. K-Nearest Neighbors (KNN)

- **Description:** A non-parametric, instance-based classifier that predicts the class of a data point based on the majority class among its k nearest neighbors.
- **Files:** `knn.py`, `main.py`

### 2. Linear Regression

- **Description:** A supervised learning algorithm for modeling the relationship between a dependent variable and one or more independent variables.
- **Files:** `linear.py`, `main.py`

### 3. Logistic Regression

- **Description:** A supervised learning algorithm for binary classification that uses the sigmoid function and gradient descent to minimize binary cross-entropy loss.
- **Files:** `logistic.py`, `main.py`

### 4. Support Vector Machine (SVM)

- **Description:** A supervised learning algorithm for binary classification that finds the optimal hyperplane to separate classes using hinge loss and gradient descent.
- **Files:** `svm.py`, `main.py`

---

## Getting Started

Each directory contains its own README and example script. To run an example:

```bash
cd KNN
python main.py
```

Replace `KNN` with any algorithm directory.

---

## Requirements

- Python 3.x
- NumPy
- scikit-learn (for generating example data and metrics)

Install dependencies with:

```bash
pip install numpy scikit-learn
```

OR:

```bash
    pip install -r requirements.txt
```

---

## License

This project is licensed under the [MIT License](LICENSE). See the LICENSE file for details.
