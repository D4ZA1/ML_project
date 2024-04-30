from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Load data
data = np.loadtxt("kinematic_features.txt")
new_data=np.loadtxt("new_features.txt")
X = data
y = np.concatenate((np.zeros(41), np.ones(55)))

# Initialize classifiers
tree = DecisionTreeClassifier()

#perform training and testing of the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=4)
tree.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = tree.predict(X_test)
y_new_pred=tree.predict(new_data.reshape(1, -1))  # Reshape new_data to 2D array
print(y_new_pred)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print results
print("Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")