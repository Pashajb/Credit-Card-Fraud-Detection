import pandas as pd

INPUT_PATH = "data/raw/creditcard.csv"
OUTPUT_PATH = "data/processed/creditcard_clean.csv"

df = pd.read_csv(INPUT_PATH)

print("Original Dataset:", df.shape)

df_clean = df.drop_duplicates().copy()

print("Clean Dataset:", df_clean.shape)

print("\nClass Distribution:")
print(df_clean["Class"].value_counts())

df_clean.to_csv(OUTPUT_PATH, index=False)

print("\nClean dataset saved successfully!")