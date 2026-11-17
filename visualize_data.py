import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/processed/creditcard_clean.csv"

df = pd.read_csv(DATA_PATH)

sns.set_theme(style="whitegrid")

# 1. Class Distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Class")
plt.title("Normal vs Fraudulent Transactions")
plt.xlabel("Transaction Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.savefig("class_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

# 2. Transaction Amount Distribution
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="Class", y="Amount")
plt.yscale("log")
plt.title("Transaction Amount by Class")
plt.xlabel("Transaction Class")
plt.ylabel("Transaction Amount (Log Scale)")
plt.savefig("amount_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

# 3. Transaction Time Distribution
plt.figure(figsize=(10, 5))
sns.histplot(
    data=df,
    x="Time",
    hue="Class",
    bins=50,
    element="step",
    stat="density",
    common_norm=False
)
plt.title("Transaction Time Distribution")
plt.xlabel("Time (Seconds)")
plt.ylabel("Density")
plt.savefig("time_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

print("All charts saved successfully!")