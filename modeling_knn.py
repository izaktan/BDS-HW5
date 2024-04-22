import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            distances = np.sqrt(np.sum((self.X_train - X.iloc[i]) ** 2, axis=1))
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train.iloc[nearest_indices]
            most_common = nearest_labels.mode()[0]
            predictions.append(most_common)
        return np.array(predictions)

    def score(self, X, y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy, predictions

def do_knn():
    raw_df = pd.read_csv("./preprocessed/reduced_dataset.csv")
    X = raw_df.drop(['Status'], axis=1)
    y = raw_df['Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    knn = KNNClassifier(k=3)
    knn.fit(X_train, y_train)

    train_acc, predictions = knn.score(X_train, y_train)
    test_acc, predictions = knn.score(X_test, y_test)

    print("KNN")
    print("Train acc:", train_acc)
    print("Test acc:", test_acc)

    tp = sum((y_test.values == 1) & (predictions == 1))
    fp = sum((y_test.values == 0) & (predictions == 1))
    fn = sum((y_test.values == 1) & (predictions == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Precision: {precision:.5f}")
    print(f"Recall: {recall:.5f}")
    print(f"F1-Score: {f1:.5f}")

# Run the function
do_knn()
