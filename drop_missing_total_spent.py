import pandas as pd

# Input and output files
CSV_IN = "Deliverable1Dataset.csv"   
CSV_OUT = "Deliverable1Dataset_drop_missing_totalspent.csv"

# Column to check
COL = "Total Spent"

# Load dataset
# Read the CSV file into a DataFrame
# pd.read_csv loads the data from CSV_IN to df
df = pd.read_csv(CSV_IN)

# Count missing before drop
# df[COL] accesses the target column
# .isna() creates a boolean Series where True is NaN
# .sum() counts the number of True values
# missingValueBefore stores the number of rows before dropping
missingValueBefore = df[COL].isna().sum()
# Print out how many rows have missing Total Spent
print(f"Rows with missing {COL} before drop: {missingValueBefore}")

# Drop rows where Total Spent is NaN
# .dropna removes rows with NaN values
# subset=[COL] contains the column which is checked for NaN
# df is reassigned to the cleaned DataFrame
df = df.dropna(subset=[COL])

# Count missing after drop
# df[COL] accesses the target column
# .isna() creates a boolean Series where True is NaN
# .sum() counts the number of True values
# missingValueAfter stores the number of rows after dropping
missingValueAfter = df[COL].isna().sum()
# Print out how many rows have missing Total Spent
print(f"Rows with missing {COL} after drop: {missingValueAfter}")

# Save cleaned dataset
df.to_csv(CSV_OUT, index=False)
print("Saved cleaned dataset:", CSV_OUT)