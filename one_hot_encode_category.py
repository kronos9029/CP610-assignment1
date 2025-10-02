# File: encode_category_onehot.py
# Task: One-Hot Encode the "Category" column
# Group: PhatPT19

import pandas as pd

# Input and output file paths
CSV_IN = "Deliverable1Dataset_item_encoded.csv"       # dataset after Item LOO encoding
CSV_OUT = "Deliverable1Dataset_category_encoded.csv"

CATEGORY = "Category"

# Load dataset
df = pd.read_csv(CSV_IN)

# One-Hot Encode Category (drop_first=False to keep all categories)
# prefix="cat" adds "cat_" prefix to each new column
# drop_first=False keeps all categories (no dummy variable trap concern for tree models)
categoryDummies = pd.get_dummies(df[CATEGORY], prefix="cat", drop_first=False, dtype=int)

# join back to the original dataset (keep original Category column if you want)
# axis=1 to join as new columns
df = pd.concat([df, categoryDummies], axis=1)

# Save output
df.to_csv(CSV_OUT, index=False)

print(f"Saved dataset with Category One-Hot Encoding: {CSV_OUT}")
print("Added columns:", list(categoryDummies.columns))
