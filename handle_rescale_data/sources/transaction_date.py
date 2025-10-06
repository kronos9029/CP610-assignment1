"""
Transaction Date Conversion and Min-Max Normalization

Pipeline Position: Run FIRST (independent variable, requires preprocessing)
Input: handle_missing_data/output_data/4_discount_applied/final_cleaned_dataset.csv
Output: handle_rescale_data/output_data/transaction_date/data_rescaling_norm_transaction_date.csv

Process:
1. Load cleaned dataset
2. Convert Transaction Date (string) to Unix timestamp (int64 seconds)
3. Apply Min-Max Normalization to [0, 1] range
4. Validate: range check, no NaN, chronological order preserved
5. Save output with original + numeric + scaled columns

Scaling Method: Min-Max Normalization
Formula: (X - X_min) / (X_max - X_min)
Range: [0, 1]
Interpretation:
  - 0 = Earliest transaction in dataset
  - 1 = Latest transaction in dataset
  - 0.5 = Midpoint of data collection period

Why Min-Max for Transaction Date?
- Dates have clear temporal bounds (min/max in dataset)
- No outliers in temporal data (all dates are valid)
- Interpretable [0, 1] scale suitable for ML algorithms
- Preserves chronological ordering and relative temporal distances
"""

import pandas as pd
import numpy as np
from pathlib import Path


# Configuration - use paths relative to this script's location
SCRIPT_DIR = Path(__file__).parent
INPUT_CSV = SCRIPT_DIR / '../../handle_missing_data/output_data/4_discount_applied/final_cleaned_dataset.csv'
OUTPUT_DIR = SCRIPT_DIR / '../output_data/transaction_date'
DATE_COLUMN = 'Transaction Date'
DATE_NUMERIC_COLUMN = 'Transaction Date Numeric'
DATE_SCALED_COLUMN = 'Transaction Date Scaled'


def convert_date_to_unix_timestamp(date_series: pd.Series) -> pd.Series:
    """
    Convert string dates to Unix timestamp (seconds since epoch).

    Args:
        date_series: Pandas Series with string dates (YYYY-MM-DD format)

    Returns:
        Pandas Series with Unix timestamps (int64 seconds)
    """
    return pd.to_datetime(date_series).astype(np.int64) // 10**9


def min_max_normalize(numeric_series: pd.Series) -> pd.Series:
    """
    Apply Min-Max Normalization to scale values to [0, 1].

    Formula: (X - X_min) / (X_max - X_min)

    Args:
        numeric_series: Pandas Series with numeric values

    Returns:
        Pandas Series with normalized values in [0, 1] range
    """
    min_val = numeric_series.min()
    max_val = numeric_series.max()
    return (numeric_series - min_val) / (max_val - min_val)


def validate_scaled_date(
    original_series: pd.Series,
    numeric_series: pd.Series,
    scaled_series: pd.Series
) -> dict:
    """
    Validate the scaled transaction date column.

    Args:
        original_series: Original string dates
        numeric_series: Unix timestamp (seconds)
        scaled_series: Normalized values [0, 1]

    Returns:
        Dictionary with validation results
    """
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

    # Overall pass
    results['all_passed'] = (
        results['range_check'] and
        results['no_missing'] and
        results['no_infinite'] and
        results['order_preserved']
    )

    return results


def process_transaction_date_rescaling(
    input_csv: Path,
    output_csv: Path,
    date_col: str = "Transaction Date",
    date_numeric_col: str = "Transaction Date Numeric",
    date_scaled_col: str = "Transaction Date Scaled"
) -> pd.DataFrame:
    """
    Complete pipeline for Transaction Date conversion and normalization.

    Args:
        input_csv: Path to input CSV file (cleaned dataset)
        output_csv: Path to output CSV file (with scaled column)
        date_col: Name of original date column (string)
        date_numeric_col: Name for Unix timestamp column (to be created)
        date_scaled_col: Name for scaled column (to be created)

    Returns:
        DataFrame with original + numeric + scaled date columns
    """
    print("="*70)
    print("TRANSACTION DATE CONVERSION AND MIN-MAX NORMALIZATION")
    print("="*70)

    # Step 1: Load dataset
    print(f"\n[1/6] Loading dataset from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"      Loaded: {len(df)} rows, {len(df.columns)} columns")
    print(f"      Original date dtype: {df[date_col].dtype}")

    # Step 2: Convert to Unix timestamp
    print(f"\n[2/6] Converting '{date_col}' to Unix timestamp...")
    df[date_numeric_col] = convert_date_to_unix_timestamp(df[date_col])
    min_timestamp = df[date_numeric_col].min()
    max_timestamp = df[date_numeric_col].max()
    print(f"      Earliest: {pd.to_datetime(min_timestamp, unit='s').date()} ({min_timestamp})")
    print(f"      Latest:   {pd.to_datetime(max_timestamp, unit='s').date()} ({max_timestamp})")
    print(f"      Span:     {(max_timestamp - min_timestamp) / (365.25 * 24 * 3600):.2f} years")

    # Step 3: Apply Min-Max Normalization
    print(f"\n[3/6] Applying Min-Max Normalization to [0, 1]...")
    df[date_scaled_col] = min_max_normalize(df[date_numeric_col])
    print(f"      Min (scaled): {df[date_scaled_col].min():.10f}")
    print(f"      Max (scaled): {df[date_scaled_col].max():.10f}")
    print(f"      Mean (scaled): {df[date_scaled_col].mean():.4f}")

    # Step 4: Validation
    print(f"\n[4/6] Validating scaled data...")
    validation = validate_scaled_date(df[date_col], df[date_numeric_col], df[date_scaled_col])
    print(f"       Range check [0, 1]: {validation['range_check']}")
    print(f"       No missing values: {validation['no_missing']}")
    print(f"       No infinite values: {validation['no_infinite']}")
    print(f"       Order preserved: {validation['order_preserved']} (corr={validation['correlation']:.10f})")
    print(f"       All checks passed: {validation['all_passed']}")

    if not validation['all_passed']:
        raise ValueError("Validation failed! Check the results above.")

    # Step 5: Save output
    print(f"\n[5/6] Saving output to: {output_csv}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"      Saved: {len(df)} rows, {len(df.columns)} columns")

    # Step 6: Summary
    print(f"\n[6/6] Summary:")
    print(f"      Input:  {input_csv.name}")
    print(f"      Output: {output_csv.name}")
    print(f"      New columns added:")
    print(f"        - {date_numeric_col}: Unix timestamp (seconds)")
    print(f"        - {date_scaled_col}: Min-Max normalized [0, 1]")

    print(f"\n{'='*70}")
    print("TRANSACTION DATE RESCALING COMPLETE!")
    print(f"{'='*70}\n")

    return df


def main():
    """
    Main function to run the Transaction Date rescaling pipeline.
    """
    # Use configuration constants
    OUTPUT_CSV = OUTPUT_DIR / 'data_rescaling_norm_transaction_date.csv'

    # Run pipeline
    df = process_transaction_date_rescaling(
        input_csv=INPUT_CSV,
        output_csv=OUTPUT_CSV,
        date_col=DATE_COLUMN,
        date_numeric_col=DATE_NUMERIC_COLUMN,
        date_scaled_col=DATE_SCALED_COLUMN
    )

    # Display sample
    print("\nSample of transformed data:")
    print(df[[DATE_COLUMN, DATE_NUMERIC_COLUMN, DATE_SCALED_COLUMN]].head(10))


if __name__ == "__main__":
    main()
