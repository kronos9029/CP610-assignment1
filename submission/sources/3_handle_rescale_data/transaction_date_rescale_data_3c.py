import pandas as pd
import numpy as np
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
INPUT_CSV = SCRIPT_DIR / '../../output/1_handle_missing_data/final_cleaned_dataset.csv'
OUTPUT_DIR = SCRIPT_DIR / '../../output/3_handle_rescale_data'
DATE_COLUMN = 'Transaction Date'
DATE_NUMERIC_COLUMN = 'Transaction Date Numeric'
DATE_SCALED_COLUMN = 'Transaction Date Scaled'

# Convert string dates to Unix timestamp (seconds since epoch).
def convert_date_to_unix_timestamp(date_series):

    return pd.to_datetime(date_series).astype(np.int64) // 10**9

# Apply Min-Max Normalization to scale values to [0, 1].
# Pandas Series with normalized values in [0, 1] range
def min_max_normalize(numeric_series):
    min_val = numeric_series.min()
    max_val = numeric_series.max()
    return (numeric_series - min_val) / (max_val - min_val)


# Validate the scaled transaction date column.
# Dictionary with validation results
def validate_scaled_date(numeric_series,scaled_series):
    results = {}

    # 1. Range check: all values in [0, 1]
    results['range_check'] = (scaled_series.min() >= 0) and (scaled_series.max() <= 1)
    results['min_value'] = scaled_series.min()
    results['max_value'] = scaled_series.max()

    # 2. No missing values
    results['no_missing'] = scaled_series.isna().sum() == 0
    results['nan_count'] = scaled_series.isna().sum()

    # 3. No infinite values
    results['no_infinite'] = np.isfinite(scaled_series).all()

    # 4. Chronological order preserved (perfect correlation)
    results['correlation'] = numeric_series.corr(scaled_series)
    results['order_preserved'] = results['correlation'] == 1.0

    # 5. Row count unchanged
    results['row_count'] = len(scaled_series)

    results['all_passed'] = (
        results['range_check'] and
        results['no_missing'] and
        results['no_infinite'] and
        results['order_preserved']
    )

    return results

# Complete pipeline for Transaction Date conversion and normalization.
# DataFrame with original + numeric + scaled date columns
def process_transaction_date_rescaling(
    input_csv,
    output_csv,
    date_col = "Transaction Date",
    date_numeric_col = "Transaction Date Numeric",
    date_scaled_col = "Transaction Date Scaled"
):

    df = pd.read_csv(input_csv)
    print(f"Loaded: {len(df)} rows, {len(df.columns)} columns")
    print(f"Original date dtype: {df[date_col].dtype}")

    # Step 2: Convert to Unix timestamp
    df[date_numeric_col] = convert_date_to_unix_timestamp(df[date_col])
    min_timestamp = df[date_numeric_col].min()
    max_timestamp = df[date_numeric_col].max()
    print(f"Earliest: {pd.to_datetime(min_timestamp, unit='s').date()} ({min_timestamp})")
    print(f"Latest: {pd.to_datetime(max_timestamp, unit='s').date()} ({max_timestamp})")
    print(f"Span: {(max_timestamp - min_timestamp) / (365.25 * 24 * 3600):.2f}")

    df[date_scaled_col] = min_max_normalize(df[date_numeric_col])
    print(f"Min (scaled): {df[date_scaled_col].min():.10f}")
    print(f"Max (scaled): {df[date_scaled_col].max():.10f}")
    print(f"Mean (scaled): {df[date_scaled_col].mean():.4f}")

    validate_scaled_date(df[date_numeric_col], df[date_scaled_col])


    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)


    return df


def main():
    OUTPUT_CSV = OUTPUT_DIR / 'data_rescaling_norm_transaction_date.csv'

    df = process_transaction_date_rescaling(
        input_csv=INPUT_CSV,
        output_csv=OUTPUT_CSV,
        date_col=DATE_COLUMN,
        date_numeric_col=DATE_NUMERIC_COLUMN,
        date_scaled_col=DATE_SCALED_COLUMN
    )

    print(df[[DATE_COLUMN, DATE_NUMERIC_COLUMN, DATE_SCALED_COLUMN]].head(10))
    print("Recommendation: Min-Max Normalization")

if __name__ == "__main__":
    main()
