from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def do_modeling(max_depth, max_features, X_train, X_test, y_train, y_test):
    gb_classifier = RandomForestClassifier(max_depth = max_depth, max_features = max_features)
    gb_classifier.fit(X_train, y_train)
    acc = gb_classifier.score(X_test, y_test)
    return acc

if __name__ == "__main__":
    data = pd.read_csv('./preprocessed/reduced_dataset.csv')

    # Split data into features and target
    X = data.drop('Status', axis=1)
    y = data['Status']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Hyperparameter grid for Random Forest
    model_grid = {
        'max_features': [1,3,5,7,10],
        'max_depth': [None, 3, 5, 10],
    }

    for max_depth in model_grid["max_depth"]:
        x_axis = []
        y_axis = []
        for max_features in model_grid["max_features"]:
            acc = do_modeling(max_depth, max_features, X_train, X_test, y_train, y_test)
            x_axis.append(max_features)
            y_axis.append(acc)
        plt.plot(x_axis, y_axis, label="max_depth="+str(max_depth))
    plt.xlabel("max_features")
    plt.ylabel("accuracy")
    plt.title("Grid Search - Random Forest")
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig('Grid Search - Random Forest')  # Save the figure to a file
    plt.close()



