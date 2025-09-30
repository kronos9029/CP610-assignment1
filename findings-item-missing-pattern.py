import pandas as pd

# Input file after Price Per Unit imputation
CSV_IN = "Deliverable1Dataset_price_imputed.csv"

# Define columns name
ITEM = "Item"
TOTAL_SPENT = "Total Spent"
PRICE_PER_UNIT = "Price Per Unit"
QUANTITY = "Quantity"
CATEGORY = "Category"
PAYMENT_METHOD = "Payment Method"
LOCATION = "Location"

# Define error
COERCE_ERRORS = "coerce"

# Load dataset
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
# missingValue is the number of rows where ITEM is NaN
# df[ITEM] accesses the target column
# .isna() creates a boolean Series where True is NaN
# .sum() counts the number of True values
missingValue = df[ITEM].isna().sum()
# missingPercent is the percentage of rows where ITEM is NaN
# missingValue / totalRow computes the fraction of missing values
missingPercent = round(missingValue / totalRow * 100, 2)

print(f"Total rows: {totalRow}")
print(f"Missing {ITEM}: {missingValue} ({missingPercent}%)")

# heck co-missingness with numeric columns
# (df[ITEM].isna()) & (df[PRICE_PER_UNIT].isna()) creates a boolean Series where both are NaN
# .sum() counts the number of True values
missingWithPrice = ((df[ITEM].isna()) & (df[PRICE_PER_UNIT].isna())).sum()
# (df[ITEM].isna()) & (df[QUANTITY].isna()) creates a boolean Series where both are NaN
# .sum() counts the number of True values
missingWithQty   = ((df[ITEM].isna()) & (df[QUANTITY].isna())).sum()
# (df[ITEM].isna()) & (df[TOTAL_SPENT].isna()) creates a boolean Series where both are NaN
# .sum() counts the number of True values
missingWithTotal = ((df[ITEM].isna()) & (df[TOTAL_SPENT].isna())).sum()

print(f"Missing both Item and Price: {missingWithPrice}")
print(f"Missing both Item and Quantity: {missingWithQty}")
print(f"Missing both Item and Total Spent: {missingWithTotal}")

# Missingness by Category
# .groupby(CATEGORY)[ITEM] groups Item by Category
# .apply applies the lambda function to Category type group of Item
# lambda s: is a lambda function where x is each group of Category type group of Item
# s.isna() creates a boolean Series where NaN = True
# .mean() computes the mean of the boolean Series
# .sort_values(ascending=False) sorts the results in descending order
# now byCategory stores the percentage of missing ITEM by CATEGORY
# multiplying by 100 converts the fraction to percentage
by_category = df.groupby(CATEGORY)[ITEM].apply(lambda s: s.isna().mean()).sort_values(ascending=False) * 100
print("\nMissing Item by Category (%):")
print(by_category.round(2))

# Missingness by Payment Method
# .groupby(PAYMENT_METHOD)[ITEM] groups Item by Category
# .apply applies the lambda function to Payment Method type group of Item
# lambda s: is a lambda function where x is each group of Payment Method type group of Item
# s.isna() creates a boolean Series where NaN = True
# .mean() computes the mean of the boolean Series
# .sort_values(ascending=False) sorts the results in descending order
# now byCategory stores the percentage of missing ITEM by PAYMENT_METHOD
# multiplying by 100 converts the fraction to percentage
by_payment = df.groupby(PAYMENT_METHOD)[ITEM].apply(lambda s: s.isna().mean()).sort_values(ascending=False) * 100
print("\nMissing Item by Payment Method (%):")
print(by_payment.round(2))

# Missingness by Location
# .groupby(LOCATION)[ITEM] groups Item by Location
# .apply applies the lambda function to each Payment Method type group of Item
# lambda s: is a lambda function where x is each group of Location type of Item
# s.isna() creates a boolean Series where NaN = True
# .mean() computes the mean of the boolean Series
# .sort_values(ascending=False) sorts the results in descending order
# now byPayment stores the percentage of missing ITEM by LOCATION
# multiplying by 100 converts the fraction to percentage
by_location = df.groupby(LOCATION)[ITEM].apply(lambda s: s.isna().mean()).sort_values(ascending=False) * 100
print("\nMissing Item by Location (%):")
print(by_location.round(2))
