from knn import KNN


def main():
    training_data = [[1, 2], [2, 3], [3, 4], [6, 7], [7, 8]]
    training_labels = ["A", "A", "A", "B", "B"]

    model = KNN(k=3)
    model.fit(training_data, training_labels)
    print(model.predict([[4, 5], [10, 20]]))


if __name__ == "__main__":
    main()
