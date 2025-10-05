import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


# Configuration
INPUT_CSV = Path('../../handle_missing_data/output_data/4_discount_applied/final_cleaned_dataset.csv')
OUTPUT_DIR = Path('../output_data/price_per_unit')
PRICE_PER_UNIT_COLUMN = 'Price Per Unit'


# Load the cleaned dataset from the missing data handling phase
# Returns a copy of the dataframe ready for rescaling operations
def load_cleaned_dataset_for_rescaling(input_csv_path):
    original_dataframe = pd.read_csv(input_csv_path)
    working_dataframe = original_dataframe.copy()

    print(f"Dataset loaded successfully")
    print(f"Shape: {working_dataframe.shape}")
    print(f"Rows: {len(working_dataframe)}")

    return working_dataframe


# Perform exploratory data analysis on Price Per Unit attribute
# Calculates descriptive statistics, detects outliers using IQR method
# Returns tuple of (Q1, Q3, IQR, outlier_count, skewness)
def analyze_price_per_unit_distribution_and_outliers(dataframe):
    print("=" * 80)
    print("PRICE PER UNIT - DESCRIPTIVE STATISTICS")
    print("=" * 80)
    print(f"\n{dataframe[PRICE_PER_UNIT_COLUMN].describe()}")

    print(f"\nMissing values: {dataframe[PRICE_PER_UNIT_COLUMN].isna().sum()}")
    print(f"Unique values: {dataframe[PRICE_PER_UNIT_COLUMN].nunique()}")
    print(f"Price range: ${dataframe[PRICE_PER_UNIT_COLUMN].min():.2f} - ${dataframe[PRICE_PER_UNIT_COLUMN].max():.2f}")

    # Outlier detection using IQR method
    Q1 = dataframe[PRICE_PER_UNIT_COLUMN].quantile(0.25)
    Q3 = dataframe[PRICE_PER_UNIT_COLUMN].quantile(0.75)
    IQR = Q3 - Q1

    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR

    outliers = dataframe[(dataframe[PRICE_PER_UNIT_COLUMN] < lower_fence) | (dataframe[PRICE_PER_UNIT_COLUMN] > upper_fence)]
    outlier_count = len(outliers)
    outlier_percentage = (outlier_count / len(dataframe)) * 100

    print("\n" + "=" * 80)
    print("OUTLIER DETECTION (IQR Method)")
    print("=" * 80)
    print(f"Q1 (25th percentile): ${Q1:.2f}")
    print(f"Q3 (75th percentile): ${Q3:.2f}")
    print(f"IQR: ${IQR:.2f}")
    print(f"Lower fence: ${lower_fence:.2f}")
    print(f"Upper fence: ${upper_fence:.2f}")
    print(f"\nNumber of outliers: {outlier_count} ({outlier_percentage:.2f}%)")

    if outlier_count > 0:
        print(f"Outlier price range: ${outliers[PRICE_PER_UNIT_COLUMN].min():.2f} to ${outliers[PRICE_PER_UNIT_COLUMN].max():.2f}")
        print(f"\nNote: Outliers were handled during Price Per Unit reconstruction (Total Spent - Quantity)")

    # Distribution analysis
    skewness = dataframe[PRICE_PER_UNIT_COLUMN].skew()
    print(f"\nDistribution Analysis:")
    print(f"- Skewness: {skewness:.4f}")
    if abs(skewness) < 0.5:
        print("Approximately symmetric distribution")
    elif skewness > 0:
        print("Right-skewed distribution (some high-priced items)")
    else:
        print("Left-skewed distribution")
    print(f"- Natural bounds: Prices > $0 with practical upper limit")
    print(f"- Outliers already handled during reconstruction phase")

    return Q1, Q3, IQR, outlier_count, skewness


# Apply Min-Max Normalization to Price Per Unit attribute (RECOMMENDED METHOD)
# Formula: (X - X_min) / (X_max - X_min)
# Returns dataframe with new 'PricePerUnit_Normalized' column in range [0, 1]
def apply_min_max_normalization_to_price_per_unit(dataframe):
    min_max_scaler = MinMaxScaler()
    dataframe['PricePerUnit_Normalized'] = min_max_scaler.fit_transform(dataframe[[PRICE_PER_UNIT_COLUMN]])


    print("MIN-MAX NORMALIZATION")
    print(f"\nOriginal range: [${dataframe[PRICE_PER_UNIT_COLUMN].min():.2f}, ${dataframe[PRICE_PER_UNIT_COLUMN].max():.2f}]")
    print(f"Scaled range: [{dataframe['PricePerUnit_Normalized'].min():.6f}, {dataframe['PricePerUnit_Normalized'].max():.6f}]")
    print(f"\nInterpretation:")
    print(f"  - 0.0 = Cheapest item (${dataframe[PRICE_PER_UNIT_COLUMN].min():.2f})")
    print(f"  - 1.0 = Most expensive item (${dataframe[PRICE_PER_UNIT_COLUMN].max():.2f})")
    print(f"  - 0.5 H Mid-range price (${(dataframe[PRICE_PER_UNIT_COLUMN].min() + dataframe[PRICE_PER_UNIT_COLUMN].max())/2:.2f})")
    print(f"\nStatistics:")
    print(dataframe['PricePerUnit_Normalized'].describe())

    return dataframe


# Apply Z-Score Standardization to Price Per Unit attribute
# Formula: (X - �) / �
# Returns dataframe with new 'PricePerUnit_Standardized' column (mean=0, std=1)
def apply_z_score_standardization_to_price_per_unit(dataframe):
    standard_scaler = StandardScaler()
    dataframe['PricePerUnit_Standardized'] = standard_scaler.fit_transform(dataframe[[PRICE_PER_UNIT_COLUMN]])

    print("Z-SCORE STANDARDIZATION")
    print(f"\nOriginal mean: ${dataframe[PRICE_PER_UNIT_COLUMN].mean():.2f}")
    print(f"Original std: ${dataframe[PRICE_PER_UNIT_COLUMN].std():.2f}")
    print(f"\nScaled mean: {dataframe['PricePerUnit_Standardized'].mean():.6f}")
    print(f"Scaled std: {dataframe['PricePerUnit_Standardized'].std():.6f}")
    print(f"\nStatistics:")
    print(dataframe['PricePerUnit_Standardized'].describe())

    return dataframe


# Apply Robust Scaling to Price Per Unit attribute
# Formula: (X - median) / IQR
# Returns dataframe with new 'PricePerUnit_Robust' column (median=0, IQR=1)
def apply_robust_scaling_to_price_per_unit(dataframe):
    robust_scaler = RobustScaler()
    dataframe['PricePerUnit_Robust'] = robust_scaler.fit_transform(dataframe[[PRICE_PER_UNIT_COLUMN]])

    IQR = dataframe[PRICE_PER_UNIT_COLUMN].quantile(0.75) - dataframe[PRICE_PER_UNIT_COLUMN].quantile(0.25)


    print("ROBUST SCALING")
    print(f"\nOriginal median: ${dataframe[PRICE_PER_UNIT_COLUMN].median():.2f}")
    print(f"Original IQR: ${IQR:.2f}")
    print(f"\nScaled median: {dataframe['PricePerUnit_Robust'].median():.6f}")

    scaled_IQR = dataframe['PricePerUnit_Robust'].quantile(0.75) - dataframe['PricePerUnit_Robust'].quantile(0.25)
    print(f"Scaled IQR: {scaled_IQR:.6f}")
    print(f"\nStatistics:")
    print(dataframe['PricePerUnit_Robust'].describe())

    return dataframe


# Compare all three rescaling methods with statistical summary
# Displays side-by-side comparison table of min, max, mean, median, std for each method
def compare_all_rescaling_methods(dataframe):
    comparison_df = pd.DataFrame({
        'Method': ['Original', 'Normalization (RECOMMENDED)', 'Standardization', 'Robust Scaling'],
        'Column': [PRICE_PER_UNIT_COLUMN, 'PricePerUnit_Normalized', 'PricePerUnit_Standardized', 'PricePerUnit_Robust'],
        'Min': [
            dataframe[PRICE_PER_UNIT_COLUMN].min(),
            dataframe['PricePerUnit_Normalized'].min(),
            dataframe['PricePerUnit_Standardized'].min(),
            dataframe['PricePerUnit_Robust'].min()
        ],
        'Max': [
            dataframe[PRICE_PER_UNIT_COLUMN].max(),
            dataframe['PricePerUnit_Normalized'].max(),
            dataframe['PricePerUnit_Standardized'].max(),
            dataframe['PricePerUnit_Robust'].max()
        ],
        'Mean': [
            dataframe[PRICE_PER_UNIT_COLUMN].mean(),
            dataframe['PricePerUnit_Normalized'].mean(),
            dataframe['PricePerUnit_Standardized'].mean(),
            dataframe['PricePerUnit_Robust'].mean()
        ],
        'Median': [
            dataframe[PRICE_PER_UNIT_COLUMN].median(),
            dataframe['PricePerUnit_Normalized'].median(),
            dataframe['PricePerUnit_Standardized'].median(),
            dataframe['PricePerUnit_Robust'].median()
        ],
        'Std': [
            dataframe[PRICE_PER_UNIT_COLUMN].std(),
            dataframe['PricePerUnit_Normalized'].std(),
            dataframe['PricePerUnit_Standardized'].std(),
            dataframe['PricePerUnit_Robust'].std()
        ]
    })

    print("COMPARISON OF ALL METHODS")
    print(comparison_df.to_string(index=False))


# Analyze interpretability of normalized prices for business understanding
# Shows how actual prices map to normalized values for clear interpretation
def analyze_price_interpretability(dataframe):
    print("INTERPRETABILITY ANALYSIS")

    # Sample prices and their scaled values
    sample_prices = [
        dataframe[PRICE_PER_UNIT_COLUMN].min(),
        dataframe[PRICE_PER_UNIT_COLUMN].quantile(0.25),
        dataframe[PRICE_PER_UNIT_COLUMN].median(),
        dataframe[PRICE_PER_UNIT_COLUMN].quantile(0.75),
        dataframe[PRICE_PER_UNIT_COLUMN].max()
    ]

    print("\nPrice Normalized Value Mapping:")
    print("-" * 80)
    for price in sample_prices:
        normalized = (price - dataframe[PRICE_PER_UNIT_COLUMN].min()) / (dataframe[PRICE_PER_UNIT_COLUMN].max() - dataframe[PRICE_PER_UNIT_COLUMN].min())
        print(f"${price:.2f} - {normalized:.4f}")



# Save all three rescaled datasets to separate CSV files
# Creates output directory if needed and saves norm, std, robust versions
def save_all_rescaled_datasets(dataframe, output_directory):
    # Create output directory if it doesn't exist
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save all three versions
    output_files = {
        'normalization': output_path / 'data_rescaling_norm_price_per_unit.csv',
        'standardization': output_path / 'data_rescaling_std_price_per_unit.csv',
        'robust': output_path / 'data_rescaling_robust_price_per_unit.csv'
    }

    # Save normalized version (recommended)
    dataframe.to_csv(output_files['normalization'], index=False)

    # Save standardized version
    dataframe.to_csv(output_files['standardization'], index=False)

    # Save robust scaled version
    dataframe.to_csv(output_files['robust'], index=False)

# Validate all rescaled columns for correctness and data integrity
# Checks ranges, missing values, row count preservation, correlation, and mathematical consistency
def validate_rescaled_data_quality(dataframe):
    print("VALIDATION CHECKS")

    # 1. Range validation
    print("\n1. RANGE VALIDATION:")
    norm_in_range = (dataframe['PricePerUnit_Normalized'] >= 0).all() and (dataframe['PricePerUnit_Normalized'] <= 1).all()
    print(f"Normalization in [0, 1]: {norm_in_range}")

    std_min = dataframe['PricePerUnit_Standardized'].min()
    std_max = dataframe['PricePerUnit_Standardized'].max()
    print(f"Standardization range: [{std_min:.2f}, {std_max:.2f}]")

    robust_min = dataframe['PricePerUnit_Robust'].min()
    robust_max = dataframe['PricePerUnit_Robust'].max()
    print(f"Robust scaling range: [{robust_min:.2f}, {robust_max:.2f}]")

    # 2. Missing values check
    print("\n2. MISSING VALUES CHECK:")
    print(f"Normalized missing: {dataframe['PricePerUnit_Normalized'].isna().sum()}")
    print(f"Standardized missing: {dataframe['PricePerUnit_Standardized'].isna().sum()}")
    print(f"Robust missing: {dataframe['PricePerUnit_Robust'].isna().sum()}")

    # 3. Row count preservation
    print("\n3. ROW COUNT PRESERVATION:")
    expected_rows = 11971
    actual_rows = len(dataframe)
    print(f"Expected: {expected_rows} rows")
    print(f"Actual: {actual_rows} rows")
    print(f"Match: {actual_rows == expected_rows}")

    # 4. Correlation with original
    print("\n4. CORRELATION WITH ORIGINAL (Should be 1.0):")
    corr_norm = dataframe[PRICE_PER_UNIT_COLUMN].corr(dataframe['PricePerUnit_Normalized'])
    corr_std = dataframe[PRICE_PER_UNIT_COLUMN].corr(dataframe['PricePerUnit_Standardized'])
    corr_robust = dataframe[PRICE_PER_UNIT_COLUMN].corr(dataframe['PricePerUnit_Robust'])

    print(f"Normalization correlation: {corr_norm:.6f}")
    print(f"Standardization correlation: {corr_std:.6f}")
    print(f"Robust correlation: {corr_robust:.6f}")

    # 5. Mathematical consistency check (Total Spent = Price � Quantity)
    print("\n5. MATHEMATICAL CONSISTENCY:")
    if 'Quantity' in dataframe.columns and 'Total Spent' in dataframe.columns:
        reconstructed = dataframe[PRICE_PER_UNIT_COLUMN] * dataframe['Quantity']
        consistency = (abs(dataframe['Total Spent'] - reconstructed) < 0.01).all()
        print(f"Total Spent = Price Per Unit - Quantity: {consistency}")
    else:
        print("Cannot verify (Quantity or Total Spent column not found)")



def main():
    working_data = load_cleaned_dataset_for_rescaling(INPUT_CSV)

    Q1, Q3, IQR, outlier_count, skewness = analyze_price_per_unit_distribution_and_outliers(working_data)

    working_data = apply_min_max_normalization_to_price_per_unit(working_data)
    working_data = apply_z_score_standardization_to_price_per_unit(working_data)
    working_data = apply_robust_scaling_to_price_per_unit(working_data)

    compare_all_rescaling_methods(working_data)

    analyze_price_interpretability(working_data)

    save_all_rescaled_datasets(working_data, OUTPUT_DIR)    # Step 8: Validate results
    validate_rescaled_data_quality(working_data)


if __name__ == "__main__":
    main()
