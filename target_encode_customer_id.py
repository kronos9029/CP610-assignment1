import pandas as pd
import numpy as np

# Input and output file paths
CSV_IN = "Deliverable1Dataset_item_imputed.csv"       
CSV_OUT = "Deliverable1Dataset_customer_id_encoded.csv"   
CUSTOMER_ID = "Customer ID"                       # categorical column to encode
TARGET_COL = "Total Spent"                            # numeric target
ENCODED_COL = "Customer ID Target Encoded"        # new encoded feature name

# Load dataset
df = pd.read_csv(CSV_IN)  # read data from CSV

# Ensure target is numeric (coerce errors to NaN to avoid math issues)
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")  # convert target to numeric

# Compute global mean of the target (used as a safe fallback)
globalMean = df[TARGET_COL].mean() # overall average of the target

# Aggregate sum and count of the target per Customer ID (used to compute leave-one-out means)
joinCustomerTotalSpent = (
    df.groupby(CUSTOMER_ID)[TARGET_COL]
      .agg(sumTotalSpentPerCus="sum", countTotalSpentPerCus="count") # sum and count of target per customer ID
)

# Join the aggregation back to the original rows (align per Customer ID)
df = df.join(joinCustomerTotalSpent, on=CUSTOMER_ID) # add sumTotalSpentPerCus and countTotalSpentPerCus to each row (align per Customer ID)

# Compute the leave-one-out numerator: group sum minus the current row's target
numeratorSeries = df["sumTotalSpentPerCus"] - df[TARGET_COL] # group sum minus this row's target

# Compute the leave-one-out denominator: group count minus the current row
denominatorSeries = df["countTotalSpentPerCus"] - 1  # group count minus this row

# Compute LOO mean; if denominatorSeries == 0 (customer appears once), result becomes NaN
with np.errstate(divide="ignore", invalid="ignore"):  # ignore divide-by-zero warnings
    looMean = numeratorSeries / denominatorSeries  # leave-one-out mean per row

# Replace NaNs with the global mean
# safe fallback for unseen/rare cases
looMean = looMean.fillna(globalMean)  

# Create the encoded feature (keep original Customer ID column)
df[ENCODED_COL] = looMean  # add new encoded column to the dataframe

# Drop helper columns to keep the output tidy
df = df.drop(columns=["sumTotalSpentPerCus", "countTotalSpentPerCus"])

# Save the result
df.to_csv(CSV_OUT, index=False)  # write the dataframe with LOO feature to CSV

print(f"Saved LOO-encoded file: {CSV_OUT}")
print(f"Encoded column: {ENCODED_COL} (fallback global mean = {globalMean})")