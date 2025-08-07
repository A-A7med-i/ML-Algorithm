from sklearn.model_selection import train_test_split
from linear import LinearRegression
from sklearn import datasets


def main():
    X, y = datasets.make_regression(
        n_samples=100, n_features=4, noise=20, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    regressor = LinearRegression(30, 0.1)
    regressor.fit(X_train, y_train)


if __name__ == "__main__":
    main()
