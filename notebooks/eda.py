"""
EDA script — run as `python notebooks/eda.py`.
Personalize this: add your own commentary on what each plot reveals,
pick a color palette you like, add 1-2 extra plots of your choosing
(e.g. age-vs-cholesterol scatter colored by target) so it isn't a
carbon copy of a template.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.preprocess import load_data, clean_data, NUMERIC_FEATURES

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "screenshots")
os.makedirs(OUT_DIR, exist_ok=True)

df = clean_data(load_data(os.path.join(os.path.dirname(__file__), "..", "data", "heart.csv")))

sns.set_theme(style="whitegrid")

# 1. Class balance
plt.figure(figsize=(5, 4))
sns.countplot(x="target", data=df)
plt.title("Class Balance: Heart Disease Presence")
plt.xlabel("Target (0 = No Disease, 1 = Disease)")
plt.savefig(f"{OUT_DIR}/class_balance.png", dpi=150, bbox_inches="tight")
plt.close()

# 2. Histograms of numeric features
df[NUMERIC_FEATURES].hist(figsize=(12, 8), bins=20)
plt.suptitle("Distribution of Numeric Features")
plt.savefig(f"{OUT_DIR}/histograms.png", dpi=150, bbox_inches="tight")
plt.close()

# 3. Correlation heatmap
plt.figure(figsize=(9, 7))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig(f"{OUT_DIR}/correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"EDA plots saved to {OUT_DIR}/")
print(df.describe(include="all").T)
