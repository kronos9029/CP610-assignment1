import pandas as pd
from pathlib import Path


# Configuration
INPUT_CSV = '../../datasource/Deliverable1Dataset.csv'
OUTPUT_CSV = '../output_data/1_total_spent/total_spent_cleaned.csv'

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


# Load original dataset and create working copy
# Returns dataframe ready for Total Spent analysis
def load_original_dataset(input_csv_path):
    original_dataframe = pd.read_csv(input_csv_path)
    working_dataframe = original_dataframe.copy()


    return working_dataframe


# Convert numeric columns to proper data types and quantify missing Total Spent
# Returns tuple of (total_rows, missing_count, missing_percentage)
def quantify_missing_total_spent(dataframe):
    # Convert columns to numeric, coercing errors to NaN
    for col in [PRICE_PER_UNIT, QUANTITY, TOTAL_SPENT]:
        dataframe[col] = pd.to_numeric(dataframe[col], errors=COERCE_ERRORS)

    total_row = len(dataframe)
    missing_value = dataframe[TOTAL_SPENT].isna().sum()
    missing_percent = round(missing_value / total_row * 100, 2)

    print('MISSING TOTAL SPENT ANALYSIS')
    print(f'Total rows: {total_row}')
    print(f'Missing {TOTAL_SPENT}: {missing_value} ({missing_percent}%)')

    return total_row, missing_value, missing_percent


# Analyze missingness patterns across categories, payment methods, and locations
# Returns tuple of (category_summary, payment_summary, location_summary)
def analyze_missingness_mechanism(dataframe):
    print('MISSINGNESS MECHANISM ANALYSIS')

    # Missingness by Category
    by_category = dataframe.groupby(CATEGORY)[TOTAL_SPENT].apply(lambda x: x.isna().mean()).sort_values(ascending=False) * 100
    print('\nMissing Total Spent by Category (%):')
    print(by_category.round(2))

    # Missingness by Payment Method
    by_payment = dataframe.groupby(PAYMENT_METHOD)[TOTAL_SPENT].apply(lambda x: x.isna().mean()).sort_values(ascending=False) * 100
    print('\nMissing Total Spent by Payment Method (%):')
    print(by_payment.round(2))

    # Missingness by Location
    by_location = dataframe.groupby(LOCATION)[TOTAL_SPENT].apply(lambda x: x.isna().mean()).sort_values(ascending=False) * 100
    print('\nMissing Total Spent by Location (%):')
    print(by_location.round(2))

    return by_category, by_payment, by_location


# Check co-missingness with Quantity, Price Per Unit, and Item fields
# Returns tuple of (qty_overlap, price_overlap, item_overlap, item_overlap_percentage)
def analyze_co_missingness_patterns(dataframe, missing_value):
    print('CO-MISSINGNESS ANALYSIS')

    # Related missingness with Quantity
    relate_missing_qty = ((dataframe[TOTAL_SPENT].isna()) & (dataframe[QUANTITY].isna())).sum()
    print(f'Rows with both Total Spent and Quantity missing: {relate_missing_qty}')
    print(f'Total Spent missing: {missing_value}')
    print(f'Perfect overlap: {relate_missing_qty == missing_value}')

    # Check co-missingness with Price Per Unit
    relate_missing_price = ((dataframe[TOTAL_SPENT].isna()) & (dataframe[PRICE_PER_UNIT].isna())).sum()
    print(f'\nRows with both Total Spent and Price Per Unit missing: {relate_missing_price}')

    # Check co-missingness with Item
    related_missing_item = ((dataframe[TOTAL_SPENT].isna()) & (dataframe[ITEM].isna())).sum()
    item_overlap_pct = related_missing_item / missing_value * 100 if missing_value > 0 else 0
    print(f'Rows with both Total Spent and Item missing: {related_missing_item}')
    print(f'Percentage of Total Spent missing cases with Item also missing: {item_overlap_pct:.2f}%')

    return relate_missing_qty, relate_missing_price, related_missing_item, item_overlap_pct


# Assess how many missing Total Spent values can be reconstructed using formula
# Returns tuple of (reconstructable_count, reconstruction_rate, irrecoverable_count, irrecoverable_rate)
def assess_reconstructability(dataframe, missing_value):
    print('RECONSTRUCTABILITY ASSESSMENT')

    # Check if Total Spent can be reconstructed from Price Per Unit and Quantity
    reconstructable = dataframe[TOTAL_SPENT].isna() & dataframe[PRICE_PER_UNIT].notna() & dataframe[QUANTITY].notna()
    reconstructable_count = reconstructable.sum()
    reconstruction_rate = reconstructable_count / missing_value if missing_value > 0 else 0

    print(f'Missing Total Spent that CAN be reconstructed: {reconstructable_count} out of {missing_value}')
    print(f'Reconstruction rate: {reconstruction_rate:.1%}')

    # Check irrecoverable cases (missing Total Spent AND at least one other field)
    irrecoverable = dataframe[TOTAL_SPENT].isna() & ~reconstructable
    irrecoverable_count = irrecoverable.sum()
    irrecoverable_rate = irrecoverable_count / missing_value if missing_value > 0 else 0

    print(f'\nMissing Total Spent that CANNOT be reconstructed: {irrecoverable_count} out of {missing_value}')
    print(f'Irrecoverable rate: {irrecoverable_rate:.1%}')

    return reconstructable_count, reconstruction_rate, irrecoverable_count, irrecoverable_rate


# Display sample of rows with missing Total Spent for inspection before deletion
def display_missing_sample(dataframe):
    print('SAMPLE OF ROWS WITH MISSING TOTAL SPENT')

    print(dataframe[dataframe[TOTAL_SPENT].isna()][[TRANSACTION_ID, CATEGORY, ITEM, PRICE_PER_UNIT, QUANTITY, TOTAL_SPENT]].head(10))


# Perform listwise deletion of rows with missing Total Spent
# Returns tuple of (cleaned_dataframe, rows_before, rows_after, retention_rate)
def perform_listwise_deletion(dataframe, missing_value):
    print('PERFORMING LISTWISE DELETION')

    rows_before = len(dataframe)
    print(f'Rows before deletion: {rows_before}')
    print(f'Rows to be deleted: {missing_value}')

    # Perform listwise deletion
    data_cleaned = dataframe.dropna(subset=[TOTAL_SPENT])

    rows_after = len(data_cleaned)
    retention_rate = (rows_after / rows_before) * 100

    print(f'\nRows after deletion: {rows_after}')
    print(f'Rows deleted: {rows_before - rows_after}')
    print(f'Data retention rate: {retention_rate:.2f}%')

    return data_cleaned, rows_before, rows_after, retention_rate


# Verify side benefits of deletion on other columns with missing values
# Returns tuple of (total_spent_complete, quantity_complete)
def validate_deletion_side_benefits(dataframe):
    print('VALIDATION - SIDE BENEFITS OF DELETION')

    print('Missing value counts after Total Spent deletion:')

    for col in [TOTAL_SPENT, QUANTITY, PRICE_PER_UNIT, ITEM]:
        missing_count = dataframe[col].isna().sum()
        missing_pct = (missing_count / len(dataframe)) * 100
        print(f'{col:20s}: {missing_count:5d} ({missing_pct:5.2f}%)')

    total_spent_complete = dataframe[TOTAL_SPENT].isna().sum() == 0
    quantity_complete = dataframe[QUANTITY].isna().sum() == 0

    print(f'\nTotal Spent is now 100% complete: {total_spent_complete}')
    print(f'Quantity is now 100% complete: {quantity_complete}')

    return total_spent_complete, quantity_complete


# Verify mathematical consistency: Total Spent = Price Per Unit x Quantity
# Returns tuple of (complete_rows_count, inconsistent_count, consistency_rate)
def validate_mathematical_consistency(dataframe):
    print('MATHEMATICAL CONSISTENCY CHECK')

    # Create filter for rows where all three numeric fields are present
    complete_rows = dataframe[[PRICE_PER_UNIT, QUANTITY, TOTAL_SPENT]].notna().all(axis=1)
    complete_data = dataframe[complete_rows].copy()

    print(f'Rows with complete Price, Quantity, and Total Spent: {len(complete_data)}')

    # Calculate expected Total Spent using the formula
    complete_data['Calculated Total'] = complete_data['Price Per Unit'] * complete_data['Quantity']

    # Calculate absolute difference between actual and calculated
    complete_data['Difference'] = abs(complete_data['Total Spent'] - complete_data['Calculated Total'])

    # Count rows with significant differences (> 0.01 to account for floating point precision)
    inconsistent = (complete_data['Difference'] > 0.01).sum()

    consistency_rate = ((len(complete_data) - inconsistent) / len(complete_data) * 100) if len(complete_data) > 0 else 0
    print(f'Rows with mathematical inconsistency (diff > 0.01): {inconsistent}')
    print(f'Mathematical consistency rate: {consistency_rate:.2f}%')

    if inconsistent == 0:
        print('\nAll rows satisfy the formula: Total Spent = Price Per Unit x Quantity')
    else:
        print(f'\nWarning: {inconsistent} rows have inconsistent calculations')
        print('\nSample of inconsistent rows:')
        print(complete_data[complete_data['Difference'] > 0.01][[PRICE_PER_UNIT, QUANTITY, TOTAL_SPENT, 'Calculated Total', 'Difference']].head())

    return len(complete_data), inconsistent, consistency_rate


# Analyze impact of deletion on remaining missing values (Item and Price Per Unit)
# Returns tuple of (item_removed, item_remaining, price_removed, price_remaining)
def analyze_impact_on_remaining_missing_values(dataframe_before, dataframe_after):
    print('IMPACT ON REMAINING MISSING VALUES')

    print('Impact on Item missingness:')
    item_missing_before = dataframe_before[ITEM].isna().sum()
    item_missing_after = dataframe_after[ITEM].isna().sum()
    item_removed = item_missing_before - item_missing_after
    item_removed_pct = (item_removed / item_missing_before * 100) if item_missing_before else 0

    print(f'Item missing before: {item_missing_before}')
    print(f'Item missing after: {item_missing_after}')
    print(f'Item missing rows removed: {item_removed} ({item_removed_pct:.1f}% of Item missing cases)')
    print(f'Item missing rows remaining: {item_missing_after} ({(item_missing_after / item_missing_before * 100) if item_missing_before else 0:.1f}%)')

    print('\nImpact on Price Per Unit missingness:')
    price_missing_before = dataframe_before[PRICE_PER_UNIT].isna().sum()
    price_missing_after = dataframe_after[PRICE_PER_UNIT].isna().sum()
    price_removed = price_missing_before - price_missing_after
    price_removed_pct = (price_removed / price_missing_before * 100) if price_missing_before else 0

    print(f'Price Per Unit missing before: {price_missing_before}')
    print(f'Price Per Unit missing after: {price_missing_after}')
    print(f'Price Per Unit missing rows removed: {price_removed} ({price_removed_pct:.1f}%)')
    print(f'Price Per Unit missing rows remaining: {price_missing_after} ({(price_missing_after / price_missing_before * 100) if price_missing_before else 0:.1f}%)')

    print(f'  - {price_missing_after} Price Per Unit values can be reconstructed using Total / Quantity')
    print(f'  - {item_missing_after} Item values can be imputed using Category information')

    return item_removed, item_missing_after, price_removed, price_missing_after


# Save the cleaned dataset with Total Spent missing rows removed to CSV file
# Creates output directory if it doesn't exist
def save_total_spent_cleaned_dataset(dataframe, output_csv_path):
    # Create output directory if it doesn't exist
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the dataset
    dataframe.to_csv(output_path, index=False)



def main():
    working_data = load_original_dataset(INPUT_CSV)

    total_row, missing_value, missing_percent = quantify_missing_total_spent(working_data)

    by_category, by_payment, by_location = analyze_missingness_mechanism(working_data)

    qty_overlap, price_overlap, item_overlap, item_overlap_pct = analyze_co_missingness_patterns(working_data, missing_value)

    reconstructable_count, reconstruction_rate, irrecoverable_count, irrecoverable_rate = assess_reconstructability(working_data, missing_value)

    display_missing_sample(working_data)

    data_cleaned, rows_before, rows_after, retention_rate = perform_listwise_deletion(working_data, missing_value)

    total_spent_complete, quantity_complete = validate_deletion_side_benefits(data_cleaned)

    complete_rows_count, inconsistent, consistency_rate = validate_mathematical_consistency(data_cleaned)

    item_removed, item_remaining, price_removed, price_remaining = analyze_impact_on_remaining_missing_values(working_data, data_cleaned)

    save_total_spent_cleaned_dataset(data_cleaned, OUTPUT_CSV)


if __name__ == "__main__":
    main()
