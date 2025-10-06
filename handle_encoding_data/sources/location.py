import pandas as pd
from pathlib import Path


# Configuration - use paths relative to this script's location
SCRIPT_DIR = Path(__file__).parent
CSV_IN = SCRIPT_DIR / "../../handle_missing_data/output_data/4_discount_applied/final_cleaned_dataset.csv"
CSV_OUT = SCRIPT_DIR / "../output_data/2_location/location_binary_encoded.csv"
LOCATION = "Location"
ENCODED_COLUMN = "Location_Encoded"


# Load the cleaned dataset and create a working copy
# Returns dataframe ready for location encoding
def load_cleaned_dataset_for_location_encoding(input_csv_path):
    original_dataframe = pd.read_csv(input_csv_path)
    working_dataframe = original_dataframe.copy()
    return working_dataframe


# Apply binary encoding to Location column
# Encoding rule: In-store = 0, Online = 1
# Returns dataframe with new Location_Encoded column
def apply_binary_encoding_to_location(dataframe):
    # Apply binary encoding: In-store = 0, Online = 1
    dataframe[ENCODED_COLUMN] = (dataframe[LOCATION] == 'Online').astype(int)

    print(f"\nOriginal Location vs Encoded:")
    print(dataframe[[LOCATION, ENCODED_COLUMN]].head(10))

    print(f"\nEncoded value distribution:")
    print(dataframe[ENCODED_COLUMN].value_counts().sort_index())

    return dataframe


# Validate binary encoding correctness through multiple checks
# Ensures: only 0/1 values, no missing values, correct mapping (In-store�0, Online�1)
def validate_binary_encoding_correctness(dataframe):
    # 1. Check only contains 0/1 values
    valid_values = dataframe[ENCODED_COLUMN].isin([0, 1]).all()
    print(f"\n1. Only contains 0/1 values: {valid_values}")

    # 2. Check no missing values
    no_missing = dataframe[ENCODED_COLUMN].isna().sum() == 0
    print(f"2. No missing values: {no_missing}")

    # 3. Verify mapping correctness
    in_store_check = (dataframe[dataframe[LOCATION] == 'In-store'][ENCODED_COLUMN] == 0).all()
    online_check = (dataframe[dataframe[LOCATION] == 'Online'][ENCODED_COLUMN] == 1).all()
    print(f"3. In-store - 0 mapping correct: {in_store_check}")
    print(f"4. Online - 1 mapping correct: {online_check}")

    # 5. Display distribution statistics
    print(f"\nDistribution statistics:")
    total_rows = len(dataframe)
    in_store_count = (dataframe[ENCODED_COLUMN] == 0).sum()
    online_count = (dataframe[ENCODED_COLUMN] == 1).sum()
    print(f"  In-store (0): {in_store_count} ({in_store_count/total_rows*100:.2f}%)")
    print(f"  Online (1): {online_count} ({online_count/total_rows*100:.2f}%)")


# Save the encoded dataset to CSV file
# Creates output directory if it doesn't exist
def save_location_encoded_dataset(dataframe, output_csv_path):
    # Create output directory if it doesn't exist
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the encoded dataset
    dataframe.to_csv(output_path, index=False)



def main():
    working_data = load_cleaned_dataset_for_location_encoding(CSV_IN)
    encoded_data = apply_binary_encoding_to_location(working_data)
    validate_binary_encoding_correctness(encoded_data)
    save_location_encoded_dataset(encoded_data, CSV_OUT)


if __name__ == "__main__":
    main()
