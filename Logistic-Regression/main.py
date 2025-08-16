from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from logistic import LogisticRegression
from sklearn.metrics import f1_score
import numpy as np


def main():
    X, y = make_classification(n_classes=2)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    lr = LogisticRegression(max_iterations=100, learning_rate=0.2)
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)

    f1 = f1_score(y_test, y_pred)
    accuracy = np.mean(y_test == y_pred)

    print(f"\nF1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
