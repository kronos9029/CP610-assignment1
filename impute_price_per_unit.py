import pandas as pd

# Input and output files
CSV_IN = "Deliverable1Dataset_drop_missing_totalspent.csv"
CSV_OUT = "Deliverable1Dataset_price_imputed.csv"

PRICE_PER_UNIT = "Price Per Unit"
TOTAL_SPENT = "Total Spent"
QUANTITY = "Quantity"

# Load dataset
# Read the CSV file into a DataFrame
# pd.read_csv loads the data from CSV_IN to df
df = pd.read_csv(CSV_IN)

# Count missing before imputation
# df[PRICE_PER_UNIT] accesses the column
# .isna() creates a boolean Series where True is NaN
# .sum() counts the number of True values
# missingValueBefore stores the number of rows before imputation
missingValueBefore = df[PRICE_PER_UNIT].isna().sum()
# Print out how many rows have missing Price Per Unit
print(f"Missing {PRICE_PER_UNIT} before: {missingValueBefore}")

# Deterministic imputation: Price = Total / Quantity
# Create a filter for rows where Price Per Unit is NaN, but Total Spent and Quantity are not NaN
filter = df[PRICE_PER_UNIT].isna() & df[TOTAL_SPENT].notna() & df[QUANTITY].notna()
# Apply the imputation formula to these rows: Price = Total / Quantity
df.loc[filter, PRICE_PER_UNIT] = df.loc[filter, TOTAL_SPENT] / df.loc[filter, QUANTITY]

# Count missing after imputation
# df[PRICE_PER_UNIT] accesses the column
# .isna() creates a boolean Series where True is NaN
# .sum() counts the number of True values
# missingValueBefore stores the number of rows before imputation
missingValueAfter = df[PRICE_PER_UNIT].isna().sum()
# Print out how many rows have missing Price Per Unit
print(f"Missing {PRICE_PER_UNIT} after: {missingValueAfter}")

# Save cleaned dataset
df.to_csv(CSV_OUT, index=False)
print("Saved dataset with imputed Price Per Unit:", CSV_OUT)
