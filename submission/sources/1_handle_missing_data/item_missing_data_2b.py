import pandas as pd
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
INPUT_CSV = SCRIPT_DIR / '../../output/1_handle_missing_data/price_per_unit_reconstructed.csv'
OUTPUT_CSV = SCRIPT_DIR / '../../output/1_handle_missing_data/item_imputed.csv'

# Column names
TOTAL_SPENT = 'Total Spent'
PRICE_PER_UNIT = 'Price Per Unit'
QUANTITY = 'Quantity'
CATEGORY = 'Category'
PAYMENT_METHOD = 'Payment Method'
LOCATION = 'Location'
ITEM = 'Item'
TRANSACTION_ID = 'Transaction ID'


# Load dataset from previous step (Price Per Unit reconstruction) and create working copy
# Returns dataframe ready for Item imputation
def load_dataset_after_price_reconstruction(input_csv_path):
    original_dataframe = pd.read_csv(input_csv_path)
    working_dataframe = original_dataframe.copy()

    return working_dataframe


# Quantify missing Item values and verify previous steps are complete
# Returns tuple of (missing_count, missing_percentage, missing_boolean_series)
def quantify_missing_item(dataframe):
    missing_item = dataframe[ITEM].isna()
    missing_count = missing_item.sum()
    missing_percentage = missing_item.mean() * 100

    print('MISSING ITEM ANALYSIS')
    print(f'Missing Item rows: {missing_count} of {len(dataframe)} ({missing_percentage:.2f}%)')

    return missing_count, missing_percentage, missing_item


# Analyze missingness patterns across categories to determine if MAR
# Returns summary of missing percentages by category, payment method, and location
def analyze_missingness_mechanism(dataframe, missing_item):
    print('MISSINGNESS ANALYSIS')

    # Analyze missingness patterns across categories
    summary = dataframe.assign(missing_item=missing_item).groupby(CATEGORY)['missing_item'].mean().sort_values(ascending=False) * 100
    print('Share of Item missing by Category:')
    print(summary.round(2).astype(str) + '%')

    # Analyze missingness patterns across payment methods
    payment_share = dataframe.assign(missing_item=missing_item).groupby(PAYMENT_METHOD)['missing_item'].mean().sort_values(ascending=False) * 100
    print('\nShare of Item missing by Payment Method:')
    print(payment_share.round(2).astype(str) + '%')

    # Analyze missingness patterns across locations
    location_share = dataframe.assign(missing_item=missing_item).groupby(LOCATION)['missing_item'].mean().sort_values(ascending=False) * 100
    print('\nShare of Item missing by Location:')
    print(location_share.round(2).astype(str) + '%')

    return summary, payment_share, location_share


# Check if all missing Item rows have valid Category information for imputation
# Returns tuple of (rows_with_both_missing, category_coverage_percentage)
def verify_category_coverage(dataframe, missing_item):
    print('CATEGORY COVERAGE ANALYSIS')

    item_and_category_missing = (missing_item & dataframe[CATEGORY].isna()).sum()
    category_coverage = ((missing_item.sum() - item_and_category_missing) / missing_item.sum() * 100) if missing_item.sum() > 0 else 0

    print(f'Rows with both Item and Category missing: {item_and_category_missing}')
    print(f'Rows with Item missing but Category present: {missing_item.sum() - item_and_category_missing}')
    print(f'Category coverage for missing Items: {category_coverage:.1f}%')


    return item_and_category_missing, category_coverage


# Analyze Item distribution within each Category to determine mode for imputation
# Returns dictionary of category to mode item mapping
def analyze_item_distribution_by_category(dataframe, missing_item):
    print('ITEM DISTRIBUTION ANALYSIS')

    category_mode_map = {}

    for category in dataframe[CATEGORY].unique():
        category_data = dataframe[(dataframe[CATEGORY] == category) & (dataframe[ITEM].notna())]

        if len(category_data) > 0:
            mode_item = category_data[ITEM].mode()[0]

            category_mode_map[category] = mode_item

    # Show unique item counts per category
    print('Item variety per Category:')
    item_variety = dataframe[dataframe[ITEM].notna()].groupby(CATEGORY)[ITEM].nunique().sort_values(ascending=False)
    for category, count in item_variety.items():
        print(f'{category:40s}: {count:3d} unique items')

    return category_mode_map


# Display sample of rows with missing Item for inspection before imputation
def display_missing_sample(dataframe, missing_item):
    print('SAMPLE OF ROWS WITH MISSING ITEM')

    print(dataframe[missing_item][[TRANSACTION_ID, CATEGORY, ITEM, PRICE_PER_UNIT, QUANTITY, TOTAL_SPENT]].head(10))



# Perform mode imputation by Category for missing Item values
# Returns dataframe with all missing Item values imputed
def impute_item_by_category_mode(dataframe, missing_item, category_mode_map):
    print('PERFORMING MODE IMPUTATION BY CATEGORY')

    item_missing_before = missing_item.sum()
    print(f'Item missing before imputation: {item_missing_before}')

    imputation_details = []

    for category in dataframe[CATEGORY].unique():
        category_mask = (dataframe[CATEGORY] == category) & missing_item
        missing_in_category = category_mask.sum()

        if missing_in_category > 0:
            category_items = dataframe[(dataframe[CATEGORY] == category) & (dataframe[ITEM].notna())][ITEM]

            if len(category_items) > 0:
                mode_item = category_items.mode()[0]
                dataframe.loc[category_mask, ITEM] = mode_item

                imputation_details.append({
                    'Category': category,
                    'Missing Count': missing_in_category,
                    'Imputed With': mode_item
                })

    item_missing_after = dataframe[ITEM].isna().sum()
    values_imputed = item_missing_before - item_missing_after

    print(f'\nItem missing after imputation: {item_missing_after}')
    print(f'Values successfully imputed: {values_imputed}')
    print(f'Imputation success rate: {values_imputed / item_missing_before:.1%}')

    print('Imputation Details by Category:')
    imputation_df = pd.DataFrame(imputation_details)
    print(imputation_df.to_string(index=False))

    return dataframe


# Verify imputation correctness and Category-Item consistency
# Returns tuple of (all_complete, consistency_issues_count)
def validate_imputation_correctness(dataframe):
    print('VALIDATION - IMPUTATION CORRECTNESS')

    print('Missing value check after imputation:')
    print(f'Item missing: {dataframe[ITEM].isna().sum()}')
    print(f'Price Per Unit missing: {dataframe[PRICE_PER_UNIT].isna().sum()}')
    print(f'Quantity missing: {dataframe[QUANTITY].isna().sum()}')
    print(f'Total Spent missing: {dataframe[TOTAL_SPENT].isna().sum()}')
    print(f'\nItem is now 100% complete: {dataframe[ITEM].isna().sum() == 0}')


# Display sample of imputed rows to verify imputation worked correctly
def display_imputed_sample(dataframe, missing_item):
    print('SAMPLE AFTER IMPUTATION')

    sample_imputed = dataframe[missing_item][[TRANSACTION_ID, CATEGORY, ITEM, PRICE_PER_UNIT, QUANTITY, TOTAL_SPENT]].head(10)
    print(sample_imputed)

    print('\nVerification by Category:')

    imputed_data = dataframe[missing_item]
    for category in imputed_data[CATEGORY].unique():
        category_imputed = imputed_data[imputed_data[CATEGORY] == category]
        imputed_item = category_imputed[ITEM].iloc[0] if len(category_imputed) > 0 else 'N/A'
        count = len(category_imputed)
        print(f'{category:40s}: {count:3d} rows imputed with "{imputed_item}"')


# Analyze current missing data status after Item imputation
def display_missing_status_after_imputation(dataframe):
    print('CURRENT MISSING VALUE STATUS')

    missing_summary = dataframe.isnull().sum()
    missing_cols = missing_summary[missing_summary > 0]

    if len(missing_cols) > 0:
        print('Columns with missing values:')
        for col, count in missing_cols.items():
            pct = (count / len(dataframe)) * 100
            print(f'{col:30s}: {count:5d} ({pct:5.2f}%)')
    else:
        print('No missing values in any column')


# Save the dataset with Item imputed to CSV file
# Creates output directory if it doesn't exist
def save_item_imputed_dataset(dataframe, output_csv_path):
    # Create output directory if it doesn't exist
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the dataset
    dataframe.to_csv(output_path, index=False)


def main():
    working_data = load_dataset_after_price_reconstruction(INPUT_CSV)
    missing_count, missing_percentage, missing_item = quantify_missing_item(working_data)

    analyze_missingness_mechanism(working_data, missing_item)

    verify_category_coverage(working_data, missing_item)

    category_mode_map = analyze_item_distribution_by_category(working_data, missing_item)

    display_missing_sample(working_data, missing_item)

    working_data = impute_item_by_category_mode(working_data, missing_item, category_mode_map)

    validate_imputation_correctness(working_data)

    display_imputed_sample(working_data, missing_item)

    display_missing_status_after_imputation(working_data)

    save_item_imputed_dataset(working_data, OUTPUT_CSV)


if __name__ == "__main__":
    main()
