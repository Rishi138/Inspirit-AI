import pandas as pd

df = pd.read_csv('cleaning_data.csv')

# Empty cells can potentially give you a wrong result when you analyze data.

# One way to deal with empty cells is to remove rows that contain empty cells.
# This is usually OK, since data sets can be very big, and removing a few rows will not have a big impact on the result.

df.dropna(inplace=True)

# Another way of dealing with empty cells is to insert a new value instead.
# This way you do not have to delete entire rows just because of some empty cells.

fill_df = df.fillna(120, inplace=True)
# Replaces all NULL values with 130

# The fillna() method allows us to replace empty cells with a value

# To only replace empty values for one column, specify the column name for the DataFrame
df["Calories"].fillna(130, inplace=True)

# Pandas uses the mean() median() and mode() methods to calculate the respective values for a specified column

df["Calories"].mean()

# Let's try to convert all cells in the 'Date' column into dates.
# Panda has a to_datetime() method for this

df['Date'] = pd.to_datetime(df['Date'])

# The result from the converting in the example above gave us a NaT value, which can be handled as a NULL value,
# and we can remove the row by using the dropna() method.

df.dropna(subset=["Date"], inplace=True)

# One way to fix wrong values is to replace them with something else.

df.loc[7, 'Duration'] = 45

# For small data sets you might be able to replace the wrong data one by one, but not for big data sets.


# To replace wrong data for larger data sets you can create some rules, e.g. set some boundaries for legal values,
# and replace any values that are outside the boundaries.
'''
for x in df.index:
  if df.loc[x, "Duration"] > 120:
    df.loc[x, "Duration"] = 120
'''

# Another way of handling wrong data is to remove the rows that contain wrong data
for x in df.index:
    if df.loc[x, "Duration"] > 120:
        df.drop(x, inplace=True)

# To discover duplicates, we can use the duplicated() method.

# The duplicated() method returns a Boolean values for each row

df.duplicated()

# To remove duplicates, use the drop_duplicates() method.

df.drop_duplicates(inplace = True)