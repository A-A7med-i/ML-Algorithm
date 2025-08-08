from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score
from SVM.svm import SVC


def main():
    X, y = make_classification(
        n_samples=200, n_features=10, n_classes=2, random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    svm_model = SVC(max_iterations=100, learning_rate=0.01, C=1.0)
    svm_model.fit(X_train, y_train)

    predictions = svm_model.predict(X_test)

    score = f1_score(y_test, predictions)
    print(f"\nF1 Score on Test Set: {score:.4f}")


if __name__ == "__main__":
    main()
