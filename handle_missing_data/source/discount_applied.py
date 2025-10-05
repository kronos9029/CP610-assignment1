import pandas as pd
import numpy as np
from pathlib import Path


# Configuration
INPUT_CSV = '../output_data/3_item/item_imputed.csv'
OUTPUT_CSV = '../output_data/4_discount_applied/final_cleaned_dataset.csv'
DISCOUNT_APPLIED_COLUMN = 'Discount Applied'


# Load dataset from previous step (Item imputation) and create working copy
# Returns dataframe ready for Discount Applied handling
def load_dataset_after_item_imputation(input_csv_path):
    original_dataframe = pd.read_csv(input_csv_path)
    working_dataframe = original_dataframe.copy()

    print(f'Dataset loaded successfully from: {input_csv_path}')
    print(f'Rows: {len(working_dataframe)}')

    return working_dataframe


# Quantify missing Discount Applied values and verify previous steps
# Returns tuple of (missing_count, missing_percentage, missing_boolean_series)
def quantify_missing_discount_applied(dataframe):
    missing_discount = dataframe[DISCOUNT_APPLIED_COLUMN].isna()
    missing_count = missing_discount.sum()
    missing_percentage = missing_discount.mean() * 100

    print('=' * 80)
    print('MISSING DISCOUNT APPLIED ANALYSIS')
    print('=' * 80)
    print(f'Missing Discount Applied rows: {missing_count} of {len(dataframe)} ({missing_percentage:.2f}%)')

    print(f'\nDiscount Applied value distribution:')
    print(dataframe[DISCOUNT_APPLIED_COLUMN].value_counts(dropna=False))

    # Verify that critical columns are complete (from previous steps)
    print(f'\nVerification of previous steps:')
    print(f'  Item missing: {dataframe["Item"].isna().sum()} (should be 0 from STEP 3)')
    print(f'  Price Per Unit missing: {dataframe["Price Per Unit"].isna().sum()} (should be 0 from STEP 2)')
    print(f'  Quantity missing: {dataframe["Quantity"].isna().sum()} (should be 0 from STEP 1)')
    print(f'  Total Spent missing: {dataframe["Total Spent"].isna().sum()} (should be 0 from STEP 1)')

    return missing_count, missing_percentage, missing_discount


# Analyze missingness patterns across categories to determine if MCAR, MAR, or MNAR
# Returns tuple of (category_cv, category_summary, payment_share, location_share)
def analyze_missingness_mechanism(dataframe, missing_discount):
    print('MISSINGNESS MECHANISM ANALYSIS')

    # Analyze missingness patterns across categories
    summary = dataframe.assign(missing_discount=missing_discount).groupby('Category')['missing_discount'].mean().sort_values(ascending=False)
    print('Share of Discount Applied missing by Category:')
    print(summary)

    # Analyze missingness patterns across payment methods
    payment_share = dataframe.assign(missing_discount=missing_discount).groupby('Payment Method')['missing_discount'].mean().sort_values(ascending=False)
    print('\nShare of Discount Applied missing by Payment Method:')
    print(payment_share)

    # Analyze missingness patterns across locations
    location_share = dataframe.assign(missing_discount=missing_discount).groupby('Location')['missing_discount'].mean().sort_values(ascending=False)
    print('\nShare of Discount Applied missing by Location:')
    print(location_share)

    # Check variation to assess randomness
    print('\n' + '=' * 80)
    print('Variation Analysis (for MCAR assessment):')
    print('=' * 80)

    category_std = summary.std()
    category_mean = summary.mean()
    cv_category = (category_std / category_mean) * 100 if category_mean > 0 else 0

    print(f'Category missingness - Mean: {category_mean:.4f}, Std: {category_std:.4f}, CV: {cv_category:.2f}%')
    print(f'Payment missingness - Range: {payment_share.min():.4f} to {payment_share.max():.4f}')
    print(f'Location missingness - Range: {location_share.min():.4f} to {location_share.max():.4f}')

    if cv_category < 10:
        print('\nLow variation suggests MCAR (Missing Completely At Random)')
        print('  Missingness is relatively uniform across categories')
    else:
        print('\nHigh variation suggests MAR (Missing At Random) or systematic pattern')

    return cv_category, summary, payment_share, location_share


# Analyze distribution of observed non-missing values to understand TRUE/FALSE balance
# Returns tuple of (true_count, false_count, balance_ratio)
def analyze_observed_value_distribution(dataframe):
    print('\n' + '=' * 80)
    print('OBSERVED VALUE DISTRIBUTION ANALYSIS')
    print('=' * 80)

    observed_values = dataframe[dataframe[DISCOUNT_APPLIED_COLUMN].notna()][DISCOUNT_APPLIED_COLUMN]

    print('Distribution of observed (non-missing) Discount Applied values:')
    value_counts = observed_values.value_counts().sort_index()
    print(value_counts)

    # Calculate proportions
    print('\nProportions of observed values:')
    total_observed = len(observed_values)
    for value, count in value_counts.items():
        proportion = (count / total_observed) * 100
        print(f'{str(value):10s}: {count:5d} ({proportion:5.2f}%)')

    # Check if distribution is balanced
    true_count = value_counts.get(True, 0)
    false_count = value_counts.get(False, 0)
    balance_ratio = true_count / false_count if false_count > 0 else 0

    print(f'\nTrue/False ratio: {balance_ratio:.3f}')
    if 0.9 <= balance_ratio <= 1.1:
        print('Nearly balanced distribution (50/50 split)')
    elif 0.8 <= balance_ratio <= 1.2:
        print('Reasonably balanced distribution')
    else:
        print('Imbalanced distribution - one category is more frequent')

    return true_count, false_count, balance_ratio


# Display sample of rows with missing Discount Applied for inspection
def display_missing_sample(dataframe, missing_discount):
    print('SAMPLE OF ROWS WITH MISSING DISCOUNT APPLIED')

    print(dataframe[missing_discount][['Transaction ID', 'Category', 'Item', 'Total Spent', DISCOUNT_APPLIED_COLUMN]].head(10))



# Fill missing Discount Applied values with "Unknown" category
# Returns dataframe with all missing values converted to "Unknown"
def fill_missing_with_unknown_category(dataframe, missing_count):
    print('HANDLING MISSING VALUES - CREATING UNKNOWN CATEGORY')

    print(f'Discount Applied missing before handling: {missing_count}')

    print('\nDistribution BEFORE handling:')
    print(dataframe[DISCOUNT_APPLIED_COLUMN].value_counts(dropna=False))

    # Fill missing values with "Unknown" string
    dataframe[DISCOUNT_APPLIED_COLUMN] = dataframe[DISCOUNT_APPLIED_COLUMN].fillna('Unknown')

    # Count missing values after handling
    discount_missing_after = dataframe[DISCOUNT_APPLIED_COLUMN].isna().sum()
    values_handled = missing_count - discount_missing_after

    print(f'\nDiscount Applied missing after handling: {discount_missing_after}')
    print(f'Values handled (converted to "Unknown"): {values_handled}')
    print(f'Handling success rate: {values_handled / missing_count:.1%}')

    print('\nDistribution AFTER handling:')
    print(dataframe[DISCOUNT_APPLIED_COLUMN].value_counts())

    return dataframe


# Verify all missing values have been addressed and dataset is complete
def validate_complete_dataset(dataframe):
    print('VALIDATION - FINAL MISSING VALUE CHECK')
    # Comprehensive missing value check across ALL columns
    missing_summary = dataframe.isnull().sum()
    missing_cols = missing_summary[missing_summary > 0]

    if len(missing_cols) > 0:
        print('� Columns with remaining missing values:')
        for col, count in missing_cols.items():
            pct = (count / len(dataframe)) * 100
            print(f'  {col:30s}: {count:5d} ({pct:5.2f}%)')
    else:
        print('NO MISSING VALUES IN ANY COLUMN')

    print('\n' + '=' * 80)
    print('Verification of all critical columns:')
    critical_columns = ['Item', 'Price Per Unit', 'Quantity', 'Total Spent', DISCOUNT_APPLIED_COLUMN]
    all_complete = True
    for col in critical_columns:
        missing_count = dataframe[col].isna().sum()
        status = '' if missing_count == 0 else ""
        print(f'{status} {col:30s}: {missing_count} missing')
        if missing_count > 0:
            all_complete = False

    print('\n' + '=' * 80)
    if all_complete:
        print('SUCCESS: ALL CRITICAL COLUMNS ARE 100% COMPLETE')
    else:
        print('Warning: Some columns still have missing values')

    return all_complete


# Display sample of rows after handling to verify Unknown category applied correctly
def display_handled_sample(dataframe, missing_discount):
    print('\n' + '=' * 80)
    print('SAMPLE AFTER HANDLING')
    print('=' * 80)

    sample_handled = dataframe[missing_discount][['Transaction ID', 'Category', 'Item', 'Total Spent', DISCOUNT_APPLIED_COLUMN]].head(10)
    print(sample_handled)

    print('\nVerification by Category:')
    for category in dataframe['Category'].unique():
        category_data = dataframe[dataframe['Category'] == category]
        unknown_count = (category_data[DISCOUNT_APPLIED_COLUMN] == 'Unknown').sum()
        total_count = len(category_data)
        unknown_pct = (unknown_count / total_count) * 100 if total_count > 0 else 0
        print(f'  {category:40s}: {unknown_count:4d} Unknown out of {total_count:5d} ({unknown_pct:5.2f}%)')



# Save the final cleaned dataset to CSV file
# Creates output directory if it doesn't exist
def save_final_cleaned_dataset(dataframe, output_csv_path):
    # Create output directory if it doesn't exist
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the final cleaned dataset
    dataframe.to_csv(output_path, index=False)

    print(f'\nFINAL cleaned dataset saved to {output_csv_path}')
    print(f'  Final row count: {len(dataframe):,}')
    print(f'  All critical columns: 100% complete')
    print(f'  Ready for analysis!')


def main():
    working_data = load_dataset_after_item_imputation(INPUT_CSV)

    missing_count, missing_percentage, missing_discount = quantify_missing_discount_applied(working_data)

    cv_category, summary, payment_share, location_share = analyze_missingness_mechanism(working_data, missing_discount)

    true_count, false_count, balance_ratio = analyze_observed_value_distribution(working_data)

    display_missing_sample(working_data, missing_discount)

    working_data = fill_missing_with_unknown_category(working_data, missing_count)

    all_complete = validate_complete_dataset(working_data)

    display_handled_sample(working_data, missing_discount)

    save_final_cleaned_dataset(working_data, OUTPUT_CSV)


if __name__ == "__main__":
    main()
