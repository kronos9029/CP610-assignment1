import pandas as pd
from pathlib import Path


# Configuration
CSV_IN = "../../handle_missing_data/output_data/4_discount_applied/final_cleaned_dataset.csv"
CSV_OUT = "../../output_data/4_discount_applied/discount_applied_one_hot_encoded.csv"
DISCOUNT_APPLIED = "Discount Applied"
ENCODING_PREFIX = "Discount"


# Load the cleaned dataset and create a working copy
# Returns dataframe ready for discount applied encoding
def load_cleaned_dataset_for_encoding(input_csv_path):
    original_dataframe = pd.read_csv(input_csv_path)
    working_dataframe = original_dataframe.copy()
    return working_dataframe


# Apply one-hot encoding to Discount Applied column
# Creates 3 binary columns: Discount_True, Discount_False, Discount_Unknown
# Each row will have exactly one "1" and two "0"s
def apply_one_hot_encoding_to_discount_applied(dataframe):
    # Apply one-hot encoding
    discount_encoded = pd.get_dummies(
        dataframe[DISCOUNT_APPLIED],
        prefix=ENCODING_PREFIX,
        drop_first=False
    )

    # Add encoded columns to dataframe
    encoded_dataframe = pd.concat([dataframe, discount_encoded], axis=1)

    print(f"\nOriginal Discount Applied vs Encoded:")
    encoded_columns = ['Discount_False', 'Discount_True', 'Discount_Unknown']
    print(encoded_dataframe[[DISCOUNT_APPLIED] + encoded_columns].head(10))

    return encoded_dataframe


# Validate one-hot encoding correctness through multiple checks
# Ensures: each row sums to 1, no missing values, only binary 0/1 values
def validate_discount_encoding_correctness(dataframe):
    encoded_columns = ['Discount_False', 'Discount_True', 'Discount_Unknown']

    # 1. Check one-hot columns sum to 1 per row
    one_hot_sum = dataframe[encoded_columns].sum(axis=1)
    sum_check = (one_hot_sum == 1).all()
    print(f"\n1. Each row has exactly one '1' across encoded columns: {sum_check}")

    # 2. Check no missing values
    no_missing = dataframe[encoded_columns].isna().sum().sum() == 0
    print(f"2. No missing values in encoded columns: {no_missing}")

    # 3. Check only binary values (0 or 1)
    binary_check = dataframe[encoded_columns].isin([0, 1]).all().all()
    print(f"3. Only contains 0/1 values: {binary_check}")

    # 4. Display distribution
    print(f"\nDistribution of encoded columns:")
    for col in encoded_columns:
        count = dataframe[col].sum()
        percentage = (count / len(dataframe)) * 100
        print(f"  {col}: {count} ({percentage:.2f}%)")


# Save the encoded dataset to CSV file
# Creates output directory if it doesn't exist
def save_discount_encoded_dataset(dataframe, output_csv_path):
    # Create output directory if it doesn't exist
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the encoded dataset
    dataframe.to_csv(output_path, index=False)


def main():
    # Step 1: Load cleaned dataset
    working_data = load_cleaned_dataset_for_encoding(CSV_IN)

    # Step 2: Apply one-hot encoding
    encoded_data = apply_one_hot_encoding_to_discount_applied(working_data)

    # Step 3: Validate encoding
    validate_discount_encoding_correctness(encoded_data)

    # Step 4: Save encoded dataset
    save_discount_encoded_dataset(encoded_data, CSV_OUT)


if __name__ == "__main__":
    main()
