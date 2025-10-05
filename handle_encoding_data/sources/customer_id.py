import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


# Configuration
CSV_IN = Path("../../handle_missing_data/output_data/4_discount_applied/final_cleaned_dataset.csv")
CSV_OUT = Path("../output_data/1_customer_id/encoded_customer_id_dataset.csv")
CUSTOMER_ID = "Customer ID"
TARGET_COL = "Total Spent"
ENCODED_COL = "Customer ID Target Encoded"


# Load data from CSV and perform basic checks on data quality
# Ensures target column is numeric and displays overview statistics
def load_and_validate_input_data(input_csv_path):
    dataframe = pd.read_csv(input_csv_path)

    # Ensure target is numeric
    dataframe[TARGET_COL] = pd.to_numeric(dataframe[TARGET_COL], errors="coerce")

    # Quick overview
    total_rows, total_columns = dataframe.shape
    missing_values_summary = dataframe[[CUSTOMER_ID, TARGET_COL]].isnull().sum()

    print(dataframe.head(3))
    print(f"\nRows: {total_rows}, Cols: {total_columns}")
    print("\nNulls in key columns:")
    print(missing_values_summary)

    return dataframe


# Compute Leave-One-Out (LOO) target encoding for Customer ID
# For each row, calculates mean Total Spent for same Customer ID excluding current row
# This prevents target leakage by not using the row's own target value in its encoding
def compute_leave_one_out_target_encoding(dataframe, global_target_mean):
    # Aggregate sum and count of the target per Customer ID
    customer_aggregated_spending = (
        dataframe.groupby(CUSTOMER_ID)[TARGET_COL]
          .agg(sum_total_spent_per_customer="sum", count_total_spent_per_customer="count")
    )

    # Join the aggregation back to the original rows
    dataframe = dataframe.join(customer_aggregated_spending, on=CUSTOMER_ID)

    print(f"\nGlobal mean of {TARGET_COL}: {global_target_mean:.6f}")
    print(dataframe[[CUSTOMER_ID, TARGET_COL, "sum_total_spent_per_customer", "count_total_spent_per_customer"]].head(3))

    # Compute the leave-one-out numerator: group sum minus the current row's target
    loo_numerator = dataframe["sum_total_spent_per_customer"] - dataframe[TARGET_COL]

    # Compute the leave-one-out denominator: group count minus the current row
    loo_denominator = dataframe["count_total_spent_per_customer"] - 1

    # Compute LOO mean; if denominator == 0 (customer appears once), result becomes NaN
    with np.errstate(divide="ignore", invalid="ignore"):
        leave_one_out_mean = loo_numerator / loo_denominator

    # Replace NaNs with the global mean (safe fallback for unseen/rare cases)
    leave_one_out_mean = leave_one_out_mean.fillna(global_target_mean)

    # Create the encoded feature (keep original Customer ID column)
    dataframe[ENCODED_COL] = leave_one_out_mean

    # Drop helper columns to keep the output tidy
    dataframe = dataframe.drop(columns=["sum_total_spent_per_customer", "count_total_spent_per_customer"])

    print("\nEncoded feature preview:")
    print(dataframe[[CUSTOMER_ID, TARGET_COL, ENCODED_COL]].head(5))

    return dataframe


# Run diagnostic checks on encoded data to verify correctness
# Identifies singleton Customer IDs (appearing once) - their encoding should equal global mean
def validate_encoding_correctness(dataframe, global_target_mean):
    # Diagnostics: spot checks
    # IDs with single occurrence should equal global mean
    customer_id_frequency_counts = dataframe[CUSTOMER_ID].value_counts()
    singleton_customer_ids = customer_id_frequency_counts[customer_id_frequency_counts == 1].index[:5]

    if singleton_customer_ids.empty:
        print("\nNo singleton IDs found — every Customer ID appears more than once.")
    else:
        print("\nSample singleton IDs (encoded should equal global mean):", list(singleton_customer_ids))
        print(dataframe[dataframe[CUSTOMER_ID].isin(singleton_customer_ids)][[CUSTOMER_ID, TARGET_COL, ENCODED_COL]].head(10))


# Create histogram and density plot to visualize distribution of encoded values
# Shows the typical customer spending pattern captured by the encoding
def visualize_encoded_distribution(dataframe):
    plt.figure(figsize=(8, 5))
    sns.histplot(
        dataframe[ENCODED_COL],
        bins=40,
        kde=True,
        color="cornflowerblue",
        edgecolor="white"
    )
    plt.title("Histogram & Density of Customer ID Encodings (LOO)")
    plt.xlabel("Encoded Value (Mean Total Spent)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    print("\nVisualization interpretation:")
    print("- Most customers fall between about 125 and 133, with peak around 130")
    print("- This represents the 'typical' customer spend in the encoded feature")
    print("- Gentle variation on either side with few high- or low-spend outliers")


# Save the encoded dataset to CSV file
# Creates output directory if it doesn't exist and saves with encoded feature column
def save_encoded_dataset_to_csv(dataframe, output_csv_path, global_target_mean):
    # Ensure output directory exists
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(output_csv_path, index=False)
    print(f"\nSaved: {output_csv_path.resolve()}")
    print(f"New column: {ENCODED_COL} (global mean fallback = {global_target_mean:.6f})")



def main():
    # Step 1: Load and check data
    print("Step 1: Loading data and performing basic checks...")
    input_dataframe = load_and_validate_input_data(CSV_IN)

    # Step 2: Compute global mean
    global_target_mean = input_dataframe[TARGET_COL].mean()

    # Step 3: Compute LOO encoding
    print("\nStep 2: Computing Leave-One-Out encoding...")
    encoded_dataframe = compute_leave_one_out_target_encoding(input_dataframe, global_target_mean)

    # Step 4: Run diagnostics
    validate_encoding_correctness(encoded_dataframe, global_target_mean)

    # Step 5: Visualize encoding distribution
    visualize_encoded_distribution(encoded_dataframe)

    # Step 6: Save encoded dataset
    save_encoded_dataset_to_csv(encoded_dataframe, CSV_OUT, global_target_mean)\


if __name__ == "__main__":
    main()
