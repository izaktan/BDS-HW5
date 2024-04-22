import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


def evaluate_model(model, X_test, y_test, model_name):
    # Predict on the testing data
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    
    # Print the results
    print(f"{model_name} performance:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print()

# Function to plot feature importances
def plot_feature_importances(model, title):
    n_features = X_train.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X_train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.title(title)
    plt.show()
    plt.savefig(title)  # Save the figure to a file
    plt.close()

# Load the dataset
data = pd.read_csv('./preprocessed/reduced_dataset.csv')

# Split data into features and target
X = data.drop('Status', axis=1)
y = data['Status']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

nb_classifier = GaussianNB()
dt_classifier = DecisionTreeClassifier(random_state=42)
rf_classifier = RandomForestClassifier(random_state=42)
gb_classifier = GradientBoostingClassifier(random_state=42)

# Train the models
nb_classifier.fit(X_train, y_train)
dt_classifier.fit(X_train, y_train)
rf_classifier.fit(X_train, y_train)
gb_classifier.fit(X_train, y_train)

evaluate_model(nb_classifier, X_test, y_test, 'Naïve Bayes')
evaluate_model(dt_classifier, X_test, y_test, 'Decision Tree')
evaluate_model(rf_classifier, X_test, y_test, 'Random Forest')
evaluate_model(gb_classifier, X_test, y_test, 'Gradient Boosting')

# Plot feature importances for Decision Tree, Random Forest, and Gradient Boosting
plot_feature_importances(dt_classifier, "Feature Importance - Decision Tree")
plot_feature_importances(rf_classifier, "Feature Importance - Random Forest")
plot_feature_importances(gb_classifier, "Feature Importance - Gradient Boosting")