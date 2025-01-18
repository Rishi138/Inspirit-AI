import pandas as pd

df = pd.read_csv("cleaning_practice.csv")

df["Count"].fillna(df["Count"].mean(),inplace = True)
df.dropna(inplace=True)
df["Date"] = pd.to_datetime(df["Date"])
df.drop_duplicates(inplace = True)
