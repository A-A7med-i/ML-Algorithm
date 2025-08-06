# K-Nearest Neighbors (KNN) Classifier

This directory contains a simple implementation of the K-Nearest Neighbors (KNN) algorithm in Python, with the main logic in `knn.py` and an example usage in `main.py`.

---

## How KNN Works

K-Nearest Neighbors (KNN) is a supervised machine learning algorithm used for classification. It predicts the class of a new data point by looking at the 'k' closest labeled data points in the training set and assigning the most common class among them.

### Steps of the KNN Algorithm

1. **Choose the number of neighbors (k):** This is a user-defined parameter.
2. **Compute distances:** For a new data point, calculate the distance to all points in the training data. The most common metric is Euclidean distance.
3. **Find the k nearest neighbors:** Select the k training samples with the smallest distances.
4. **Vote:** Assign the class that is most common among the k neighbors.

### Mathematical Explanation

**Euclidean Distance:**

For two points $x = (x_1, x_2, ..., x_n)$ and $y = (y_1, y_2, ..., y_n)$ in n-dimensional space, the Euclidean distance is:

$$d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$

This formula is used to measure how close two points are in feature space. The k points with the smallest distances are selected as neighbors.

**Majority Voting:**

After finding the k nearest neighbors, count the frequency of each class label among them. The class with the highest count is assigned to the new data point.

---

## Files

- `knn.py`: Contains the `KNN` class with `fit` and `predict` methods.
- `main.py`: Example script showing how to use the KNN classifier.

---

## Usage

1. **Import and Initialize**

    ```python
    from knn import KNN
    model = KNN(k=3)
    ```

2. **Fit the Model**

    ```python
    training_data = [[1, 2], [2, 3], [3, 4], [6, 7], [7, 8]]
    training_labels = ["A", "A", "A", "B", "B"]
    model.fit(training_data, training_labels)
    ```

3. **Make Predictions**

    ```python
    predictions = model.predict([[4, 5], [10, 20]])
    print(predictions)  # Example output: ['A', 'B']
    ```

Or simply run the example:

```bash
python main.py
```

---

## Requirements

- Python 3.x
- NumPy

Install dependencies with:

```bash
pip install numpy
```

---
