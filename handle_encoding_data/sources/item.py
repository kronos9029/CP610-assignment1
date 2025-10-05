import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import KFold


# Configuration
CSV_IN = Path("../output_data/1_customer_id/encoded_customer_id_dataset.csv")
CSV_OUT = Path("../output_data/6_item/encoded_item_dataset.csv")

# Column names
ITEM = "Item"
TARGET_COL = "Total Spent"
ENCODED_COL = "Item Target Encoded"

# 2-Fold parameters
N_SPLITS = 2          # fixed to 2-fold
RANDOM_STATE = 42     # random seed for reproducibility
SHUFFLE = True        # shuffle before split


# Load encoded customer ID dataset from CSV
# Returns dataframe ready for item encoding
def load_encoded_customer_dataset(input_csv_path):
    dataframe = pd.read_csv(input_csv_path)
    return dataframe


# Prepare and validate data for target encoding
# Ensures target column is numeric and displays basic statistics
def prepare_and_validate_data_for_encoding(dataframe):
    # Ensure target is numeric
    # .to_numeric with errors="coerce" will convert non-numeric to NaN
    dataframe[TARGET_COL] = pd.to_numeric(dataframe[TARGET_COL], errors="coerce")

    # row/col count & null summary
    row_count, col_count = dataframe.shape
    null_summary = dataframe[[ITEM, TARGET_COL]].isnull().sum()

    # print basic info
    print(f"Rows: {row_count}, Cols: {col_count}")
    print("\nNulls in key columns:")
    print(null_summary)

    return dataframe


# Perform 2-fold target encoding for Item column
# Splits data into 2 folds, encodes each fold using means from opposite fold
# Returns dataframe with new Item Target Encoded column
def perform_two_fold_target_encoding(dataframe):
    # Initialize KFold with 2 splits, shuffling, and a random state for reproducibility
    kfold_splitter = KFold(n_splits=N_SPLITS, shuffle=SHUFFLE, random_state=RANDOM_STATE)

    # Calculate global mean of the target
    global_target_mean = dataframe[TARGET_COL].mean()

    # Prepare a Series to hold the encoded values with the same index as df
    encoded_values_series = pd.Series(index=dataframe.index, dtype=float)

    print(f"\nPerforming 2-Fold Target Encoding (Global mean fallback: {global_target_mean:.6f})...\n")

    # Perform 2-Fold target encoding
    # fold variables: train_idx, val_idx
    # start=1 to make fold count human-friendly (1, 2, ...)
    # kfold_splitter.split separates df into training and validation fold
    for fold_number, (train_indices, validation_indices) in enumerate(kfold_splitter.split(dataframe), start=1):
        # For 2-fold: 'train_indices' is the opposite fold used to compute means
        # 'validation_indices' is the fold we encode
        training_fold = dataframe.iloc[train_indices]  # opposite fold
        validation_fold = dataframe.iloc[validation_indices]  # current fold to encode

        # Calculate mean target per Item in training fold
        item_mean_mapping = training_fold.groupby(ITEM)[TARGET_COL].mean()

        # Map means to validation fold, fill NaN with global mean
        encoded_fold_values = validation_fold[ITEM].map(item_mean_mapping).fillna(global_target_mean)

        # Assign encoded values to the correct positions in the encoded Series
        encoded_values_series.iloc[validation_indices] = encoded_fold_values.values

        # Diagnostics
        unique_items_in_training = item_mean_mapping.index.nunique()
        print(f"Fold {fold_number}: opposite(train)={len(train_indices)} encode(val)={len(validation_indices)} | unique Items in opposite={unique_items_in_training}")

    # Attach encoded feature
    dataframe[ENCODED_COL] = encoded_values_series

    # Diagnostics
    print("\nEncoded summary:")
    print(dataframe[ENCODED_COL].describe())

    return dataframe


# Visualize the distribution of encoded Item values using violin plot
# Shows the spread of mean Total Spent values associated with each item
def visualize_item_encoding_distribution(dataframe):
    plt.figure(figsize=(8, 4))
    sns.violinplot(
        data=dataframe[[ENCODED_COL]],
        y=ENCODED_COL,
        inner="box",
        color="skyblue"
    )
    plt.title("Item Target Encoded Distribution (2-Fold OOF)")
    plt.ylabel("Encoded Value (Mean Total Spent)")
    plt.tight_layout()
    plt.show()




# Save the encoded dataset to CSV file
# Creates output directory if it doesn't exist
def save_item_encoded_dataset(dataframe, output_csv_path):
    # Ensure output directory exists
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Save encoded dataset
    dataframe.to_csv(output_csv_path, index=False)

    global_mean_value = dataframe[TARGET_COL].mean()
    print(f"\n Saved encoded dataset: {output_csv_path.resolve()}")
    print(f"New column: {ENCODED_COL}  |  Global mean fallback = {global_mean_value:.6f}")


def main():

    # Step 1: Load encoded customer dataset
    input_dataframe = load_encoded_customer_dataset(CSV_IN)

    # Step 2: Prepare and validate data
    prepared_dataframe = prepare_and_validate_data_for_encoding(input_dataframe)

    # Step 3: Perform 2-fold target encoding
    encoded_dataframe = perform_two_fold_target_encoding(prepared_dataframe)

    # Step 4: Visualize distribution
    visualize_item_encoding_distribution(encoded_dataframe)

    # Step 5: Save encoded dataset
    save_item_encoded_dataset(encoded_dataframe, CSV_OUT)


if __name__ == "__main__":
    main()
