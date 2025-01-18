import pandas as pd
from sklearn import metrics
from sklearn import linear_model
from sklearn.model_selection import train_test_split

# Load the data
dataframe = pd.read_csv("heart_disease.csv")

# Drop missing values
dataframe.dropna(inplace=True)

# Define features and target
X = dataframe.iloc[:, :-1]
y = dataframe["TenYearCHD"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = linear_model.LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Make predictions
y_predictions = model.predict(X_test)

# Calculate accuracy
accuracy = metrics.accuracy_score(y_test, y_predictions)

# Add predictions to the test dataframe
test_df = X_test.copy()
test_df["TenYearCHD"] = y_test
test_df["Predictions"] = y_predictions

# Print accuracy and first few rows of the test dataframe
print(accuracy)
print(test_df.head())
