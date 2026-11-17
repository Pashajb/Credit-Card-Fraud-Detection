import pandas as pd

DATA_PATH = "data/raw/creditcard.csv"

df = pd.read_csv(DATA_PATH)

print("\nDuplicate Transactions:")
print(df.duplicated().sum())

print("\nTransaction Amount Statistics:")
print(df.groupby("Class")["Amount"].describe())

print("\nAverage Transaction Amount:")
print(df.groupby("Class")["Amount"].mean())

print("\nMedian Transaction Amount:")
print(df.groupby("Class")["Amount"].median())

print("\nFraud Transactions by Time:")
print(df[df["Class"] == 1]["Time"].describe())

print("\nNormal Transactions by Time:")
print(df[df["Class"] == 0]["Time"].describe())