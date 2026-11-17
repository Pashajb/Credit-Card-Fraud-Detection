import pandas as pd

df = pd.read_csv("data/raw/creditcard.csv")

duplicates = df[df.duplicated(keep=False)]

print("\nTotal Rows Involved in Duplicates:")
print(len(duplicates))

print("\nClass Distribution Among Duplicate Rows:")
print(duplicates["Class"].value_counts())

print("\nFraud Count Before Removing Duplicates:")
print(df["Class"].value_counts())

clean_df = df.drop_duplicates()

print("\nDataset Shape After Removing Duplicates:")
print(clean_df.shape)

print("\nFraud Count After Removing Duplicates:")
print(clean_df["Class"].value_counts())