import pandas as pd
from sklearn.model_selection import KFold
from pathlib import Path
import matplotlib.pyplot as plt

# Files
CSV_IN = "Deliverable1Dataset.csv"
CSV_OUT = "target-encoded-customerid.csv"
LOOKUP = "customerid_mean_lookup.csv"
GMEAN_TXT = "customerid_global_mean.txt"

# Columns
CATEGORY_COL = "Customer ID"
TARGET = "Total Spent"

# K-Fold settings
K = 5
SHUFFLE = True
RANDOM_SEED = 42

dataFrame = pd.read_csv(CSV_IN)

# Ensure expected columns exist
missing_cols = {CATEGORY_COL, TARGET} - set(dataFrame.columns)
assert not missing_cols, f"Missing columns: {missing_cols}"

# Transfer to numeric, coerce errors to NaN
dataFrame[TARGET] = pd.to_numeric(dataFrame[TARGET], errors="coerce")

# If target missing, drop row
before = len(dataFrame)
dataFrame = dataFrame.dropna(subset=[TARGET])
print(f"Dropped {before - len(dataFrame)} rows with missing {TARGET} Total Spent value")

kFold = KFold(n_splits = K, shuffle = SHUFFLE, random_state = RANDOM_SEED)
globalMean = dataFrame[TARGET].mean()
print("Global mean:", round(globalMean, 4))

encoded = pd.Series(index=dataFrame.index, dtype=float)

for trainIdx, valIdx in kFold.split(dataFrame):
    train = dataFrame.iloc[trainIdx]
    val   = dataFrame.iloc[valIdx]

    # Per-ID mean target from TRAIN folds only
    means = train.groupby(CATEGORY_COL)[TARGET].mean().rename("targetEncode")

    # Map into validation fold; unseen IDs default to global mean
    valueTargetEncode = val[CATEGORY_COL].map(means).fillna(globalMean)

    # Store encodings in original order
    encoded.iloc[valIdx] = valueTargetEncode.values

# Attach encoded feature
dataFrame["CustomerID_TE"] = encoded

lookup = dataFrame.groupby(CATEGORY_COL)[TARGET].mean().rename("CustomerID_TE").reset_index()
lookup.to_csv(LOOKUP, index=False)
Path(GMEAN_TXT).write_text(str(globalMean))
print("Saved lookup:", LOOKUP, "| global mean file:", GMEAN_TXT)

dataFrame["CustomerID_TE"].hist(bins=30)
plt.title("Distribution of K-Fold Encoded CustomerID_TE")
plt.xlabel("Encoded value")
plt.ylabel("Count")
plt.tight_layout(); 
plt.show()