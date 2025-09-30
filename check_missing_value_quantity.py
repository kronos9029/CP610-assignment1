import pandas as pd

# Input and output files
CSV_IN = "Deliverable1Dataset_drop_missing_totalspent.csv"
CSV_OUT = "Deliverable1Dataset_quantity_checked.csv"

COL = "Quantity"

# Load dataset
# Read the CSV file into a DataFrame
# pd.read_csv loads the data from CSV_IN to df
df = pd.read_csv(CSV_IN)

# Count missing values
# df[COL] accesses the target column
# .isna() creates a boolean Series where True is NaN
# .sum() counts the number of True values
missingValue = df[COL].isna().sum()
print(f"Missing {COL}: {missingValue}")

if missingValue == 0:
    # No missing values, no action needed
    print(f"{COL} is complete. No imputation required.")
else:
    # Missing values found, imputation needed
    print(f"{COL} has {missingValue} missing values. Imputation required.")