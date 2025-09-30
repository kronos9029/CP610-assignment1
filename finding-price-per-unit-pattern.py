import pandas as pd

# Input file after dropping rows with missing Total Spent
CSV_IN = "Deliverable1Dataset_drop_missing_totalspent.csv"

# Define columns name
TOTAL_SPENT = "Total Spent"
PRICE_PER_UNIT = "Price Per Unit"
QUANTITY = "Quantity"
CATEGORY = "Category"
PAYMENT_METHOD = "Payment Method"
LOCATION = "Location"

# Define error
COERCE_ERRORS = "coerce"

# pd.read_csv loads the data from CSV_IN to data frame
df = pd.read_csv(CSV_IN)

# Convert column's values to numeric, coercing errors to NaN
# col is each of the relevant columns
for col in [PRICE_PER_UNIT, QUANTITY, TOTAL_SPENT]:
    # pd.to_numeric converts values in col to numeric, with errors coerced to NaN
    # df[col] accesses the target column
    # errors=COERCE_ERRORS specifies that parsing errors are set to NaN
    df[col] = pd.to_numeric(df[col], errors = COERCE_ERRORS)

# Basic counts
# rows is the total number of rows in df
totalRow = len(df)
# missingValue is the number of rows where PRICE_PER_UNIT is NaN
# df[PRICE_PER_UNIT] accesses the target column
# .isna() creates a boolean Series where True is NaN
# .sum() counts the number of True values
missingValue = df[PRICE_PER_UNIT].isna().sum()
# missingPercent is the percentage of rows where PRICE_PER_UNIT is NaN
# missingValue / totalRow computes the fraction of missing values
missingPercent = round(missingValue / totalRow * 100, 2)

print(f"Total rows: {totalRow}")
print(f"Missing {PRICE_PER_UNIT}: {missingValue} ({missingPercent}%)")

# Reconstructable check (Price = Total / Quantity) since Total Spent and Quantity has no missing values
filter = df[PRICE_PER_UNIT].isna() & df[QUANTITY].notna() & df[TOTAL_SPENT].notna()
reconstructable = filter.sum()
completenessPercent = round(reconstructable / max(missingValue, 1) * 100, 2)
print(f"Reconstructable missing values: {reconstructable} ({completenessPercent}%)")

# Co-missingness
# (df[PRICE_PER_UNIT].isna()) & (df[QUANTITY].isna()) creates a boolean Series where both are NaN
# .sum() counts the number of True values
missingWithQty = ((df[PRICE_PER_UNIT].isna()) & (df[QUANTITY].isna())).sum()
# (df[PRICE_PER_UNIT].isna()) & (df[TOTAL_SPENT].isna()) creates a boolean Series where both are NaN
# .sum() counts the number of True values
missingWithTotal = ((df[PRICE_PER_UNIT].isna()) & (df[TOTAL_SPENT].isna())).sum()

print(f"Missing both Price Per Unit and Quantity: {missingWithQty}")
print(f"Missing both Price Per Unit and Total Spent: {missingWithTotal}")

# Missingness by Category
# .groupby(CATEGORY)[PRICE_PER_UNIT] groups Price Per Unit by Category
# .apply applies the lambda function to Category type group of Price Per Unit
# lambda s: is a lambda function where x is each group of Category type group of Price Per Unit
# s.isna() creates a boolean Series where NaN = True
# .mean() computes the mean of the boolean Series
# .sort_values(ascending=False) sorts the results in descending order
# now byCategory stores the percentage of missing Price Per Unit by CATEGORY
# multiplying by 100 converts the fraction to percentage
byCategory = df.groupby(CATEGORY)[PRICE_PER_UNIT].apply(lambda s: s.isna().mean()).sort_values(ascending=False) * 100
print("\nMissing Price by Category (%):")
print(byCategory.round(2))

# Missingness by Payment Method
# .groupby(PAYMENT_METHOD)[PRICE_PER_UNIT] groups Price Per Unit by Payment Method
# .apply applies the lambda function to each Payment Method type group of Price Per Unit
# lambda s: is a lambda function where x is each group of Payment Method type of Price Per Unit
# s.isna() creates a boolean Series where NaN = True
# .mean() computes the mean of the boolean Series
# .sort_values(ascending=False) sorts the results in descending order
# now byPayment stores the percentage of missing PRICE_PER_UNIT by PAYMENT_METHOD
# multiplying by 100 converts the fraction to percentage
by_payment = df.groupby(PAYMENT_METHOD)[PRICE_PER_UNIT].apply(lambda s: s.isna().mean()).sort_values(ascending=False) * 100
print("\nMissing Price by Payment Method (%):")
print(by_payment.round(2))

# Missingness by Location
# .groupby(LOCATION)[PRICE_PER_UNIT] groups Price Per Unit by Location
# .apply applies the lambda function to each Payment Method type group of Price Per Unit
# lambda s: is a lambda function where x is each group of Location type of Price Per Unit
# s.isna() creates a boolean Series where NaN = True
# .mean() computes the mean of the boolean Series
# .sort_values(ascending=False) sorts the results in descending order
# now byPayment stores the percentage of missing PRICE_PER_UNIT by LOCATION
# multiplying by 100 converts the fraction to percentage
by_location = df.groupby(LOCATION)[PRICE_PER_UNIT].apply(lambda s: s.isna().mean()).sort_values(ascending=False) * 100
print("\nMissing Price by Location (%):")
print(by_location.round(2))
