import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_absolute_error as mae

dataframe = pd.read_csv('housing.csv')

dataframe["OceanNumber"] = dataframe["ocean_proximity"].replace(
    {"<1H OCEAN": 0, "INLAND": 1, "ISLAND": 2, "NEAR BAY": 3, "NEAR OCEAN": 4})
dataframe = dataframe.dropna()
x = dataframe[
    ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households",
     "median_income", "OceanNumber"]].values
y = dataframe[["median_house_value"]].values
model = LinearRegression()
model.fit(x, y)
values = np.array([-122.23, 37.88, 41, 880, 129, 322, 126, 8.3252, 3])
values = values.reshape(1, -1)
predictions = model.predict(x)
dataframe["Predictions"] = predictions
print(mae(y, predictions))
print(dataframe[["median_house_value", "Predictions"]])
