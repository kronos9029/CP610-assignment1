# File: findings-totalspent.py
# Task: Inspect missingness of "Total Spent" in the original dataset
# Group: PhatPT19

import pandas as pd

# Input dataset
CSV_IN = "Deliverable1Dataset.csv"

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
# missingValue is the number of rows where TOTAL_SPENT is NaN
# df[TOTAL_SPENT] accesses the target column
# .isna() creates a boolean Series where True is NaN
# .sum() counts the number of True values
missingValue = df[TOTAL_SPENT].isna().sum()
# missingPercent is the percentage of rows where TOTAL_SPENT is NaN
# missingValue / totalRow computes the fraction of missing values
missingPercent = round(missingValue / totalRow * 100, 2)

print(f"Total rows: {totalRow}")
print(f"Missing {TOTAL_SPENT}: {missingValue} ({missingPercent}%)")

# related missingness with Quantity and Price Per Unit
# (df[TOTAL_SPENT].isna()) & (df[QUANTITY].isna()) creates a boolean Series where both are NaN
relateMissingQty = ((df[TOTAL_SPENT].isna()) & (df[QUANTITY].isna())).sum()
# (df[TOTAL_SPENT].isna()) & (df[PRICE_PER_UNIT].isna()) creates a boolean Series where both are NaN
relateMissingPrice = ((df[TOTAL_SPENT].isna()) & (df[PRICE_PER_UNIT].isna())).sum()

# Print related missingness
print(f"Missing of Total Spent and Quantity: {relateMissingQty}")
print(f"Missing of Total Spent and Price Per Unit: {relateMissingPrice}")

# Missingness by Category
# .groupby(CATEGORY)[TOTAL_SPENT] groups Total Spent by Category
# .apply applies the lambda function to Category type group of Total Spent
# lambda s: is a lambda function where x is each group of Category type group of Total Spent
# s.isna() creates a boolean Series where NaN = True
# .mean() computes the mean of the boolean Series
# .sort_values(ascending=False) sorts the results in descending order
# now byCategory stores the percentage of missing TOTAL_SPENT by CATEGORY
# multiplying by 100 converts the fraction to percentage
byCategory = df.groupby(CATEGORY)[TOTAL_SPENT].apply(lambda x: x.isna().mean()).sort_values(ascending=False) * 100
print("\nMissing Total Spent by Category (%):")
print(byCategory.round(2))

# Missingness by Payment Method
# .groupby(PAYMENT_METHOD)[TOTAL_SPENT] groups Total Spent by Payment Method
# .apply applies the lambda function to each Payment Method type group of Total Spent
# lambda s: is a lambda function where x is each group of Payment Method type of Total Spent
# s.isna() creates a boolean Series where NaN = True
# .mean() computes the mean of the boolean Series
# .sort_values(ascending=False) sorts the results in descending order
# now byPayment stores the percentage of missing TOTAL_SPENT by PAYMENT_METHOD
# multiplying by 100 converts the fraction to percentage
byPayment = df.groupby(PAYMENT_METHOD)[TOTAL_SPENT].apply(lambda x: x.isna().mean()).sort_values(ascending=False) * 100
print("\nMissing Total Spent by Payment Method (%):")
print(byPayment.round(2))

# Missingness by Location
# .groupby(LOCATION)[TOTAL_SPENT] groups Total Spent by Location
# .apply applies the lambda function to each Location type group of Total Spent
# lambda s: is a lambda function where x is each group of Location type of Total Spent
# s.isna() creates a boolean Series where NaN = True
# .mean() computes the mean of the boolean Series
# .sort_values(ascending=False) sorts the results in descending order
# now byLocation stores the percentage of missing TOTAL_SPENT by LOCATION
# multiplying by 100 converts the fraction to percentage
byLocation = df.groupby(LOCATION)[TOTAL_SPENT].apply(lambda x: x.isna().mean()).sort_values(ascending=False) * 100
print("\nMissing Total Spent by Location (%):")
print(byLocation.round(2))
