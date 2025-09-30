# File: impute-item-mode-by-category.py
# Task: Impute missing 'Item' using Mode Imputation (Category-aware, then global)
# Group: PhatPT19

import pandas as pd

# Input (after Total Spent, Quantity, and Price Per Unit are complete)
CSV_IN  = "Deliverable1Dataset_price_imputed.csv"
CSV_OUT = "Deliverable1Dataset_item_imputed.csv"

COL_ITEM = "Item"
COL_CAT  = "Category"

# Load
df = pd.read_csv(CSV_IN)

# Counts before
missing_before = df[COL_ITEM].isna().sum()
print(f"Missing {COL_ITEM} before: {missing_before}")

# 1) Category-level mode mapping: most frequent Item per Category
cat_mode = (
    df.dropna(subset=[COL_ITEM, COL_CAT])
      .groupby(COL_CAT)[COL_ITEM]
      .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else pd.NA)
)

# Fill by Category mode where Item is missing and Category is known
mask_cat = df[COL_ITEM].isna() & df[COL_CAT].notna()
to_fill_cat = mask_cat.sum()
df.loc[mask_cat, COL_ITEM] = df.loc[mask_cat, COL_CAT].map(cat_mode)
filled_cat = to_fill_cat - df.loc[mask_cat, COL_ITEM].isna().sum()
print(f"Filled by Category mode: {filled_cat}")

# 2) Global mode fallback for any remaining missing
remaining = df[COL_ITEM].isna().sum()
if remaining > 0:
    global_mode = df[COL_ITEM].mode().iloc[0]
    df[COL_ITEM] = df[COL_ITEM].fillna(global_mode)
    print(f"Filled by Global mode: {remaining}")

# After
missing_after = df[COL_ITEM].isna().sum()
print(f"Missing {COL_ITEM} after: {missing_after}")

# Save
df.to_csv(CSV_OUT, index=False)
print("Saved:", CSV_OUT)
