# File: check_item_frequency_duplicates.py
# Task: Check Items that share the same frequency
# Group: PhatPT19

import pandas as pd

# Input: dataset after Item imputation (and after Customer ID encoding if you continue pipeline)
csvIn = "Deliverable1Dataset_customer_id_loo.csv"
itemCol = "Item"

# Load dataset
df = pd.read_csv(csvIn)

# Count frequency of each Item
itemFreq = df[itemCol].value_counts()

# Group Items by their frequency value
freqGroups = itemFreq.groupby(itemFreq).apply(lambda x: list(x.index))

# Print only frequencies that have more than one Item
print("Items that share the same frequency:\n")
for freq, items in freqGroups.items():
    if len(items) > 1:
        print(f"Frequency = {freq}, Items = {items[:10]}{'...' if len(items) > 10 else ''}")

print("\n[INFO] Total unique frequencies:", itemFreq.nunique())
print("[INFO] Total unique items:", itemFreq.shape[0])
