from sklearn import tree
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics

# Load the dataset
dataframe = pd.read_csv("winequality-red.csv")

# Split the data into features and target variable
X = dataframe.iloc[:, :-1]
y = dataframe["quality"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialize the model
model = tree.DecisionTreeClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_predictions = model.predict(X_test)

# Calculate accuracy
accuracy = metrics.accuracy_score(y_test, y_predictions)
print(f"Accuracy: {accuracy}")

# Plot the decision tree
plt.figure(figsize=(20, 10))
tree.plot_tree(model, filled=True, feature_names=X.columns, class_names=[str(i) for i in set(y)])
plt.show()
