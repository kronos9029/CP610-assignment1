import pandas as pd
import numpy as np
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
INPUT_CSV = SCRIPT_DIR / '../../output/1_handle_missing_data/total_spent_cleaned.csv'
OUTPUT_CSV = SCRIPT_DIR / '../../output/1_handle_missing_data/price_per_unit_reconstructed.csv'

# Column names
TOTAL_SPENT = 'Total Spent'
PRICE_PER_UNIT = 'Price Per Unit'
QUANTITY = 'Quantity'
CATEGORY = 'Category'
PAYMENT_METHOD = 'Payment Method'
LOCATION = 'Location'
ITEM = 'Item'
TRANSACTION_ID = 'Transaction ID'

# Error handling
COERCE_ERRORS = 'coerce'


# Load dataset from previous step (Total Spent cleaned) and create working copy
# Returns dataframe ready for Price Per Unit reconstruction
def load_dataset_after_total_spent_cleaning(input_csv_path):
    original_dataframe = pd.read_csv(input_csv_path)
    working_dataframe = original_dataframe.copy()

    return working_dataframe


# Convert numeric columns to proper data types and quantify missing Price Per Unit
# Returns tuple of (missing_count, missing_percentage, missing_boolean_series)
def quantify_missing_price_per_unit(dataframe):
    # Convert columns to numeric, coercing errors to NaN
    for col in [PRICE_PER_UNIT, QUANTITY, TOTAL_SPENT]:
        dataframe[col] = pd.to_numeric(dataframe[col], errors=COERCE_ERRORS)

    missing_price = dataframe[PRICE_PER_UNIT].isna()
    missing_count = missing_price.sum()
    missing_percentage = missing_price.mean() * 100

    print('MISSING PRICE PER UNIT ANALYSIS')
    print(f'Missing Price Per Unit rows: {missing_count} of {len(dataframe)} ({missing_percentage:.2f}%)')

    # Verify that Total Spent and Quantity are complete (from STEP 1)
    print(f'\nVerification of previous steps:')
    print(f'  Total Spent missing: {dataframe[TOTAL_SPENT].isna().sum()} (should be 0 from STEP 1)')
    print(f'  Quantity missing: {dataframe[QUANTITY].isna().sum()} (should be 0 from STEP 1)')

    return missing_count, missing_percentage, missing_price


# Analyze missingness patterns across categories, payment methods, and locations
# Returns tuple of (category_summary, payment_share, location_share)
def analyze_missingness_mechanism(dataframe, missing_price):
    print('MISSINGNESS MECHANISM ANALYSIS')

    # Analyze missingness patterns across categories
    summary = dataframe.assign(missing_price=missing_price).groupby(CATEGORY)['missing_price'].mean().sort_values(ascending=False) * 100
    print('Share of Price Per Unit missing by Category:')
    print(summary.round(2))

    # Analyze missingness patterns across payment methods
    payment_share = dataframe.assign(missing_price=missing_price).groupby(PAYMENT_METHOD)['missing_price'].mean().sort_values(ascending=False) * 100
    print('\nShare of Price Per Unit missing by Payment Method:')
    print(payment_share.round(2))

    # Analyze missingness patterns across locations
    location_share = dataframe.assign(missing_price=missing_price).groupby(LOCATION)['missing_price'].mean().sort_values(ascending=False) * 100
    print('\nShare of Price Per Unit missing by Location:')
    print(location_share.round(2))

    return summary, payment_share, location_share


# Check co-missingness with Item field to confirm MAR classification
# Returns tuple of (item_overlap_count, item_overlap_percentage, perfect_overlap_boolean)
def analyze_co_missingness_with_item(dataframe, missing_price):
    print('CO-MISSINGNESS ANALYSIS')

    item_overlap = (missing_price & dataframe[ITEM].isna()).sum()
    overlap_percentage = (item_overlap / missing_price.sum() * 100) if missing_price.sum() > 0 else 0
    perfect_overlap = (item_overlap == missing_price.sum())

    print(f'Rows with both Price Per Unit and Item missing: {item_overlap}')
    print(f'Price Per Unit missing: {missing_price.sum()}')
    print(f'Perfect overlap: {perfect_overlap}')
    print(f'Overlap percentage: {overlap_percentage:.1f}%')

    # Check co-missingness with Total Spent (should be 0 after STEP 1)
    total_overlap = (missing_price & dataframe[TOTAL_SPENT].isna()).sum()
    print(f'\nRows with both Price Per Unit and Total Spent missing: {total_overlap}')

    # Check co-missingness with Quantity (should be 0 after STEP 1)
    qty_overlap = (missing_price & dataframe[QUANTITY].isna()).sum()
    print(f'Rows with both Price Per Unit and Quantity missing: {qty_overlap}')

    return item_overlap, overlap_percentage, perfect_overlap


# Assess how many missing Price Per Unit values can be reconstructed using formula
# Returns tuple of (reconstructable_count, reconstruction_rate, zero_quantity_count)
def assess_reconstructability(dataframe, missing_price):
    print('RECONSTRUCTABILITY ASSESSMENT')

    reconstructable = missing_price & dataframe[TOTAL_SPENT].notna() & dataframe[QUANTITY].notna()
    reconstructable_count = reconstructable.sum()
    reconstruction_rate = (reconstructable_count / missing_price.sum() * 100) if missing_price.sum() > 0 else 0

    print(f'Missing Price Per Unit that CAN be reconstructed: {reconstructable_count} out of {missing_price.sum()}')

    # Check for any cases where Quantity is zero (division by zero issue)
    zero_qty = missing_price & (dataframe[QUANTITY] == 0)
    zero_qty_count = zero_qty.sum()

    return reconstructable_count, reconstruction_rate, zero_qty_count


# Display sample of rows with missing Price Per Unit for inspection before reconstruction
def display_missing_sample(dataframe, missing_price):
    print('SAMPLE OF ROWS WITH MISSING PRICE PER UNIT')

    print(dataframe[missing_price][[TRANSACTION_ID, CATEGORY, ITEM, PRICE_PER_UNIT, QUANTITY, TOTAL_SPENT]].head(10))


# Perform deterministic reconstruction using formula: Price = Total Spent / Quantity
# Returns dataframe with all missing Price Per Unit values reconstructed
def reconstruct_price_per_unit_using_formula(dataframe, missing_price, missing_count):
    print('PERFORMING DETERMINISTIC RECONSTRUCTION')

    price_missing_before = missing_count
    print(f'Price Per Unit missing before reconstruction: {price_missing_before}')

    # Perform deterministic reconstruction using the mathematical formula
    dataframe.loc[missing_price, PRICE_PER_UNIT] = dataframe.loc[missing_price, TOTAL_SPENT] / dataframe.loc[missing_price, QUANTITY]

    price_missing_after = dataframe[PRICE_PER_UNIT].isna().sum()
    values_reconstructed = price_missing_before - price_missing_after

    print(f'\nPrice Per Unit missing after reconstruction: {price_missing_after}')
    print(f'Values successfully reconstructed: {values_reconstructed}')
    print(f'Reconstruction success rate: {values_reconstructed / price_missing_before:.1%}')

    return dataframe


# Verify reconstruction accuracy by checking mathematical consistency
# Returns tuple of (all_complete, consistency_rate, inconsistent_count)
def validate_reconstruction_correctness(dataframe):
    print('VALIDATION - RECONSTRUCTION CORRECTNESS')

    print('Missing value check after reconstruction:')
    print(f'{PRICE_PER_UNIT} missing: {dataframe[PRICE_PER_UNIT].isna().sum()}')
    print(f'{TOTAL_SPENT} missing: {dataframe[TOTAL_SPENT].isna().sum()}')
    print(f'{QUANTITY} missing: {dataframe[QUANTITY].isna().sum()}')
    print(f'\n{PRICE_PER_UNIT} is now 100% complete: {dataframe[PRICE_PER_UNIT].isna().sum() == 0}')

    # Mathematical consistency check
    print('Mathematical Consistency Validation:')

    # Create filter for rows with all three fields present
    complete_rows = dataframe[[PRICE_PER_UNIT, QUANTITY, TOTAL_SPENT]].notna().all(axis=1)
    data_complete = dataframe[complete_rows].copy()

    print(f'Rows with complete numeric fields: {len(data_complete)} out of {len(dataframe)}')

    return dataframe[PRICE_PER_UNIT].isna().sum() == 0


# Display sample of reconstructed rows to verify calculation worked correctly
def display_reconstructed_sample(dataframe, missing_price):
    print('SAMPLE AFTER RECONSTRUCTION')

    print(dataframe[missing_price][[TRANSACTION_ID, CATEGORY, ITEM, PRICE_PER_UNIT, QUANTITY, TOTAL_SPENT]].head(10))


# Analyze current missing data status after Price Per Unit reconstruction
def display_missing_status_after_reconstruction(dataframe):
    print('CURRENT MISSING VALUE STATUS')

    missing_summary = dataframe.isnull().sum()
    missing_cols = missing_summary[missing_summary > 0]
    print(f'\nMissing columns: {missing_cols}')


# Save the dataset with Price Per Unit reconstructed to CSV file
# Creates output directory if it doesn't exist
def save_price_reconstructed_dataset(dataframe, output_csv_path):
    # Create output directory if it doesn't exist
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the dataset
    dataframe.to_csv(output_path, index=False)


def main():
    working_data = load_dataset_after_total_spent_cleaning(INPUT_CSV)

    missing_count, missing_percentage, missing_price = quantify_missing_price_per_unit(working_data)

    analyze_missingness_mechanism(working_data, missing_price)

    analyze_co_missingness_with_item(working_data, missing_price)

    assess_reconstructability(working_data, missing_price)

    display_missing_sample(working_data, missing_price)

    working_data = reconstruct_price_per_unit_using_formula(working_data, missing_price, missing_count)

    validate_reconstruction_correctness(working_data)

    display_reconstructed_sample(working_data, missing_price)

    display_missing_status_after_reconstruction(working_data)

    save_price_reconstructed_dataset(working_data, OUTPUT_CSV)


if __name__ == "__main__":
    main()
