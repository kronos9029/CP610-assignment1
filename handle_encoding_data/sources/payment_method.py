import pandas as pd
import os
from pathlib import Path


# Configuration
INPUT_CSV_PATH = Path('../../handle_missing_data/output_data/4_discount_applied/final_cleaned_dataset.csv')
OUTPUT_CSV_PATH = Path('../output_data/3_payment_method/encoded_payment_method_dataset.csv')
PAYMENT_METHOD_COLUMN = 'Payment Method'
ENCODING_PREFIX = 'Payment'


# Load the cleaned dataset and create a working copy
# Returns a dataframe ready for encoding operations
def load_cleaned_dataset(input_file_path):
    original_dataframe = pd.read_csv(input_file_path)
    working_dataframe = original_dataframe.copy()
    return working_dataframe


# Analyze payment method distribution and check data quality
# Displays unique values, percentage distribution, and missing value count
def analyze_payment_method_distribution(dataframe):
    print("Unique Payment Methods:")
    print(dataframe[PAYMENT_METHOD_COLUMN].value_counts())

    print("\nDistribution (%):")
    print(dataframe[PAYMENT_METHOD_COLUMN].value_counts(normalize=True) * 100)

    print(f"\nTotal unique values: {dataframe[PAYMENT_METHOD_COLUMN].nunique()}")
    print(f"Missing values: {dataframe[PAYMENT_METHOD_COLUMN].isna().sum()}")


# Apply one-hot encoding to Payment Method column
# Creates binary columns for each payment method category
# Returns encoded dataframe with columns: Payment_Cash, Payment_Credit Card, Payment_Digital Wallet
def apply_one_hot_encoding_to_payment_method(dataframe):
    # Apply one-hot encoding using pd.get_dummies
    # drop_first=False to keep all 3 columns for interpretability
    payment_method_encoded = pd.get_dummies(
        dataframe[PAYMENT_METHOD_COLUMN],
        prefix=ENCODING_PREFIX,
        drop_first=False
    )

    print(payment_method_encoded.columns.tolist())
    print("\nFirst few rows of encoded data:")
    print(payment_method_encoded.head(10))

    return payment_method_encoded


# Validate one-hot encoding correctness through multiple checks
# Ensures: row sums equal 1, only binary values, no missing values, distribution matches original
def validate_one_hot_encoding(dataframe, encoded_dataframe):
    # Validation 1: Each row should sum to exactly 1 (one payment method per transaction)
    row_sums = encoded_dataframe.sum(axis=1)
    print("Row sum validation:")
    print(f"All rows sum to 1: {(row_sums == 1).all()}")
    print(f"Min sum: {row_sums.min()}, Max sum: {row_sums.max()}")

    # Validation 2: Only binary values (0 or 1)
    print("\nBinary validation:")
    for column_name in encoded_dataframe.columns:
        unique_values = encoded_dataframe[column_name].unique()
        print(f"{column_name}: {sorted(unique_values)}")

    # Validation 3: No missing values
    print(f"\nMissing values: {encoded_dataframe.isna().sum().sum()}")

    # Validation 4: Total counts match original distribution
    print("\nDistribution check:")
    for column_name in encoded_dataframe.columns:
        original_payment_method = column_name.replace(f'{ENCODING_PREFIX}_', '')
        encoded_count = encoded_dataframe[column_name].sum()
        original_count = (dataframe[PAYMENT_METHOD_COLUMN] == original_payment_method).sum()
        match_status = encoded_count == original_count
        print(f"{original_payment_method}: Encoded={encoded_count}, Original={original_count}, Match={match_status}")


# Combine encoded columns with original dataset
# Adds the new binary payment method columns to the existing dataframe
def combine_encoded_with_original_dataset(original_dataframe, encoded_dataframe):
    combined_dataframe = pd.concat([original_dataframe, encoded_dataframe], axis=1)

    print("\nColumns added:")
    added_columns = [col for col in combined_dataframe.columns if col.startswith(f'{ENCODING_PREFIX}_')]
    print(added_columns)

    display_columns = [PAYMENT_METHOD_COLUMN] + added_columns
    print(combined_dataframe[display_columns].head(10))

    return combined_dataframe


# Save the encoded dataset to CSV file
# Creates output directory if it doesn't exist and saves the final encoded dataset
def save_encoded_dataset_to_csv(dataframe, output_file_path):
    # Create output directory if it doesn't exist
    output_directory = output_file_path.parent
    output_directory.mkdir(parents=True, exist_ok=True)

    # Save the encoded dataset
    dataframe.to_csv(output_file_path, index=False)

def main():
    working_data = load_cleaned_dataset(INPUT_CSV_PATH)

    analyze_payment_method_distribution(working_data)

    encoded_payment_data = apply_one_hot_encoding_to_payment_method(working_data)

    validate_one_hot_encoding(working_data, encoded_payment_data)

    final_encoded_dataset = combine_encoded_with_original_dataset(working_data, encoded_payment_data)

    save_encoded_dataset_to_csv(final_encoded_dataset, OUTPUT_CSV_PATH)


if __name__ == "__main__":
    main()
