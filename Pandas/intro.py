import pandas as pd

fruits = {
    "names": ["apple", "banana", "orange"],
    "color": ["red", "yellow", "orange"],
    "calories": [104, 105, 62]
}

pd_fruits = pd.DataFrame(fruits)

print(pd_fruits)

# Data frame is a table
# A Pandas Series is like a column in a table.
# It is a one-dimensional array holding data of any type.
# We can access series data using indexes

animal = ["monkey", "dog", "cat"]

pd_animal = pd.Series(animal)

print(pd_animal)
print(pd_animal[0])

# If nothing else is specified, the values are labeled with their index number. First value has indexed 0,
# second value has indexed 1 etc.
#
# This label can be used to access a specified value.

indexes = ["John", "George", "Bob"]
pd_animal = pd.Series(data=animal, index=indexes)
print(pd_animal)
print(pd_animal["John"])

# We can also do this by just passing a dictionary instead of an array

# To select only some items in the dictionary, use the index argument and specify only the items you want to
# include in the Series.

people = {
    "John Doe": "Engineer",
    "Monkey man": "Actor",
    "Foo": "Somebody"
}

series_people = pd.Series(people, index=["John Doe", "Foo"])
print(series_people)

print("\n\n\n\n\n\n\n")

# Data sets in Pandas are usually multidimensional tables, called DataFrames.
# Series is like a column, a DataFrame is the whole table.

detailed_people = {
    "Profession": ["Engineer", "Actor", "Somebody"],
    "Age": [27, 32, 98],
    "Salary": [200000, 500000, 15000]
}

df_people = pd.DataFrame(detailed_people, index=["John Doe", "Monkey man", "Foo"])
print(df_people)

# Pandas use the loc attribute to return one or more specified row(s)

print(df_people.loc["John Doe"])

# Returns first and third row

print(df_people.loc[["John Doe", "Monkey man"]])

# We can use df.to_string to get entire DataFrame

print(df_people.to_string())

# We can read csv files by using read_csv

df_customer = pd.read_csv("customers-100.csv")
print(df_customer)

# If our data is too large printing it will just show us a preview
# We can use to_string to print the full thing

# We can use to_json to read json data

df_data = pd.read_json('data.json')
print(df_data)

# One of the most used method for getting a quick overview of the DataFrame, is the head() method.
# The head() method returns the headers and a specified number of rows, starting from the top.

print(df_customer.head(10))
print(df_customer.loc[0])

# There is also a tail() method for viewing the last rows of the DataFrame.
# The tail() method returns the headers and a specified number of rows, starting from the bottom.

# The default rows for head and tail is 5

# The DataFrames object has a method called info(), that gives you more information about the data set.

print("\n\n\n\n\n\n\n\n\n\n")
print(df_customer.info())
