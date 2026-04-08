import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("testingset.csv")

# Basic cleanup
df.columns = df.columns.str.strip()
df["attack_cat"] = df["attack_cat"].astype("category")

# Optional: drop label if you're focusing on attack categories
# df = df.drop(columns=["label"])

# Identify column types
cat_cols = ["proto", "service", "state"]
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
num_cols = [c for c in num_cols if c not in ["id", "label"] and c not in cat_cols]

bin_cols = [c for c in num_cols if df[c].nunique() <= 2]
cont_cols = [c for c in num_cols if c not in bin_cols and c != "attack_cat"]

# 1) Overall class balance
attack_counts = df["attack_cat"].value_counts(dropna=False)
print(attack_counts)

plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="attack_cat", order=attack_counts.index)
plt.xticks(rotation=45, ha="right")
plt.title("Attack category counts")
plt.tight_layout()
plt.show()

# 2) Numerical features by attack class
for col in cont_cols[:10]:  # start with a subset
    plt.figure(figsize=(12, 5))
    sns.boxplot(data=df, x="attack_cat", y=col)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"{col} by attack category")
    plt.tight_layout()
    plt.show()

# 3) Group summaries for numeric columns
summary = (
    df.groupby("attack_cat")[cont_cols]
      .agg(["mean", "median", "std", "min", "max"])
)
summary.to_csv("attack_cat_numeric_summary.csv")

# 4) Categorical columns vs attack category
for col in cat_cols:
    ct = pd.crosstab(df["attack_cat"], df[col], normalize="index")
    ct.to_csv(f"{col}_by_attack_cat.csv")

    plt.figure(figsize=(12, 5))
    ct.plot(kind="bar", stacked=True, figsize=(12, 5), colormap="tab20")
    plt.title(f"{col} distribution within attack categories")
    plt.ylabel("Proportion")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()