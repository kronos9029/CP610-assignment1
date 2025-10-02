import pandas as pd
import numpy as np

# Input and output file paths
CSV_IN = "Deliverable1Dataset_customer_id_encoded.csv"
CSV_OUT = "Deliverable1Dataset_item_encoded.csv"
ITEM = "Item"
TARGET_COL = "Total Spent"
ENCODED_COL = "Item Encoded"

# Load dataset
df = pd.read_csv(CSV_IN)

# Ensure target is numeric (coerce errors to NaN to avoid math issues)
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce") # convert target to numeric

# Compute global mean of the target (used as a safe fallback)
globalMean = df[TARGET_COL].mean() # overall average of the target

# Aggregate sum and count of the target per Item (used to compute leave-one-out means)
joinItemTotalSpent = (
    df.groupby(ITEM)[TARGET_COL]
      .agg(sumTotalSpentPerItem="sum", countTotalSpentPerItem="count") # sum and count of target per Item
)

# Join the aggregation back to the original rows (align per Item)
df = df.join(joinItemTotalSpent, on=ITEM) # add sumTotalSpentPerCus and countTotalSpentPerCus to each row (align per Customer ID)

# Compute the leave-one-out numerator: group sum minus the current row's target
numerator = df["sumTotalSpentPerItem"] - df[TARGET_COL] # group sum minus this row's target

# Compute the leave-one-out denominator: group count minus the current row
denominator = df["countTotalSpentPerItem"] - 1 # group count minus this row

# Calculate LOO mean; rows with denominator == 0 (Item appears once), result becomes NaN
with np.errstate(divide="ignore", invalid="ignore"):  # ignore divide-by-zero warnings
    looMean = numerator / denominator # leave-one-out mean per row

# Replace NaNs with the global mean
# safe fallback for unseen/rare cases
looMean = looMean.fillna(globalMean) 

# create the encoded feature (keep original Item column)
df[ENCODED_COL] = looMean  # add new encoded column to the dataframe

# Drop helper columns to keep the output tidy
df = df.drop(columns=["sumTotalSpentPerItem", "countTotalSpentPerItem"])

# Save the result
df.to_csv(CSV_OUT, index=False)

print(f"Saved file with Item LOO encoding: {CSV_OUT}")
print(f"Encoded column: {ENCODED_COL} | Fallback (global mean) = {globalMean}")