import pandas as pd
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
BASE_PATH = SCRIPT_DIR / "../../output/2_handle_encoding_data"
OUTPUT_PATH = BASE_PATH / "final_fully_encoded_dataset.csv"

# Column definitions
ORIGINAL_CATEGORICAL_COLS = ['Customer ID', 'Category', 'Item', 'Payment Method', 'Location', 'Discount Applied']
PAYMENT_COLS = ['Payment_Cash', 'Payment_Credit Card', 'Payment_Digital Wallet']
DISCOUNT_COLS = ['Discount_False', 'Discount_True', 'Discount_Unknown']

# Load all 6 encoded datasets from the encoding output directory.
# Returns: dict: Dictionary of dataframes with keys matching encoding type
def load_all_encoded_datasets(base_path):
    datasets = {}

    # Load each encoded dataset
    file_mappings = {
        'customer': 'encoded_customer_id_dataset.csv',
        'item': 'encoded_item_dataset.csv',
        'category': 'encoded_category_dataset.csv',
        'location': 'location_binary_encoded.csv',
        'payment': 'encoded_payment_method_dataset.csv',
        'discount': 'discount_applied_one_hot_encoded.csv'
    }

    for key, filename in file_mappings.items():
        file_path = base_path / filename
        datasets[key] = pd.read_csv(file_path)

    return datasets

# Combine all encoded columns into a single dataframe.
# Uses category dataset as base and adds encoded columns from others.
# Return combined dataset with all encoded features
def combine_encoded_columns(datasets):
    print("COMBINING ENCODED COLUMNS")

    # Start with category dataset (contains all original columns + category encoding)
    final_df = datasets['category'].copy()

    # Add Customer ID Target Encoding
    final_df['Customer ID Target Encoded'] = datasets['customer']['Customer ID Target Encoded']

    # Add Item Target Encoding
    final_df['Item Target Encoded'] = datasets['item']['Item Target Encoded']

    # Add Location encoding
    final_df['Location_Encoded'] = datasets['location']['Location_Encoded']
    # Add Payment Method encodings (3 columns)
    for col in PAYMENT_COLS:
        final_df[col] = datasets['payment'][col]

    # Add Discount Applied encodings (3 columns)
    for col in DISCOUNT_COLS:
        final_df[col] = datasets['discount'][col]

    print(f"\nCombined dataset shape: {final_df.shape}")
    print(f"Total columns: {len(final_df.columns)}")

    return final_df

# Remove original categorical columns, keeping only numerical and encoded features.
# Dataset with only numerical and encoded columns
def drop_original_categorical_columns(dataframe, columns_to_drop):
    print("DROPPING ORIGINAL CATEGORICAL COLUMNS")

    # Drop original categorical columns
    cleaned_df = dataframe.drop(columns=columns_to_drop)

    print(f"Dropped columns: {columns_to_drop}")
    print(f"\nAfter dropping categorical columns:")
    print(f"Columns: {len(cleaned_df.columns)}")

    return cleaned_df

# Validate the final encoded dataset for correctness.
# Checks: missing values, one-hot sums, binary values, row count.
# Return True if all validation checks pass
def validate_combined_dataset(dataframe):
    print("VALIDATION CHECKS")

    all_checks_passed = True

    # 1. No missing values in encoded columns
    encoded_cols = [col for col in dataframe.columns
                   if col not in ['Transaction ID', 'Price Per Unit', 'Quantity', 'Total Spent', 'Transaction Date']]
    no_missing = dataframe[encoded_cols].isna().sum().sum() == 0
    if not no_missing:
        print(f"Missing values found: {dataframe[encoded_cols].isna().sum().sum()}")
        all_checks_passed = False

    # 2. Payment Method one-hot columns sum to 1 per row
    payment_sum_check = (dataframe[PAYMENT_COLS].sum(axis=1) == 1).all()
    print(f"2. Payment Method one-hot sums to 1 per row: {payment_sum_check}")
    if not payment_sum_check:
        print(f"Invalid row sums found")
        all_checks_passed = False

    # 3. Discount Applied one-hot columns sum to 1 per row
    discount_sum_check = (dataframe[DISCOUNT_COLS].sum(axis=1) == 1).all()
    print(f"3. Discount Applied one-hot sums to 1 per row: {discount_sum_check}")
    if not discount_sum_check:
        print(f"Invalid row sums found")
        all_checks_passed = False

    # 4. Category one-hot columns sum to 1 per row
    category_cols = [col for col in dataframe.columns if col.startswith('cat_')]
    category_sum_check = (dataframe[category_cols].sum(axis=1) == 1).all()
    print(f"4. Category one-hot sums to 1 per row: {category_sum_check}")
    if not category_sum_check:
        print(f"Invalid row sums found")
        all_checks_passed = False

    # 5. Location binary encoding has only 0/1
    binary_check = dataframe['Location_Encoded'].isin([0, 1]).all()
    print(f"5. Location encoding contains only 0/1: {binary_check}")
    if not binary_check:
        print(f"Invalid values found in Location_Encoded")
        all_checks_passed = False

    return all_checks_passed


#  Save the final encoded dataset to CSV.
#  Creates output directory if it doesn't exist.
def save_final_encoded_dataset(dataframe, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(output_path, index=False)


def main():
    datasets = load_all_encoded_datasets(BASE_PATH)

    combined_df = combine_encoded_columns(datasets)

    final_df = drop_original_categorical_columns(combined_df, ORIGINAL_CATEGORICAL_COLS)

    validate_combined_dataset(final_df)

    save_final_encoded_dataset(final_df, OUTPUT_PATH)


if __name__ == "__main__":
    main()
