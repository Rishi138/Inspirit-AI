import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Reading the csv files
dataframe = pd.read_csv("C:/Users/rajal/PycharmProjects/InspiritAI/LinearRegression/car_dekho.csv")
# Reformatting qualitative data to quantitative
dataframe["TransmissionNumber"] = dataframe['Transmission'].replace({'Manual': 1, "Automatic": 0})
dataframe["SellerNumber"] = dataframe['Seller_Type'].replace({'Dealer': 1, "Individual": 0})
dataframe["FuelNumber"] = dataframe['Fuel_Type'].replace({'CNG': 2, "Diesel": 1, "Petrol": 0})
# Creating training data for X
X = dataframe[["Age", "Kms_Driven", "TransmissionNumber", "SellerNumber", "FuelNumber"]].values
# Creating training data for Y
Y = dataframe[["Selling_Price"]].values

# Creating an instance of linear regression which will become our model
model = LinearRegression()

# Training our model
model.fit(X, Y)

# Calculating average accuracy

# Creating list with all predictions
predictions = []
for x in range(len(dataframe)):
    # Getting all x values needed to predict for given row
    Age = dataframe.iloc[x]["Age"]
    Kms_Driven = dataframe.iloc[x]["Kms_Driven"]
    TransmissionNumber = dataframe.iloc[x]["TransmissionNumber"]
    SellerNumber = dataframe.iloc[x]["SellerNumber"]
    FuelNumber = dataframe.iloc[x]["FuelNumber"]
    # Creating and reshaping x values
    values = np.array([Age, Kms_Driven, TransmissionNumber, SellerNumber, FuelNumber])
    values = values.reshape(1, -1)
    # Predicting price
    prediction = model.predict(values)[0][0]
    # Adding prediction to our array
    predictions.append(prediction)
# Adding new column called predictions with prediction in dataframe
dataframe["Predictions"] = predictions
# Creating an array for selling price
actual = np.array(dataframe["Selling_Price"].values)
# Making an array with all percentages of how accurate model is
# Ratio of 1 means perfect prediction
accuracy = actual / predictions
# Printing final average accuracy
print(sum(accuracy) / len(accuracy))

