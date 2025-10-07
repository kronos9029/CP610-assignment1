import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


SCRIPT_DIR = Path(__file__).parent
INPUT_CSV = SCRIPT_DIR / '../../output/1_handle_missing_data/final_cleaned_dataset.csv'
OUTPUT_DIR = SCRIPT_DIR / '../../output/3_handle_rescale_data'
QUANTITY_COLUMN = 'Quantity'


# Load the cleaned dataset from the missing data handling phase
# Returns a copy of the dataframe ready for rescaling operations
def load_cleaned_dataset_for_rescaling(input_csv_path):
    original_dataframe = pd.read_csv(input_csv_path)
    working_dataframe = original_dataframe.copy()

    return working_dataframe


# Perform exploratory data analysis on Quantity attribute
# Calculates descriptive statistics, detects outliers using IQR method
# Returns tuple of (Q1, Q3, IQR, outlier_count, outliers_dataframe)
def analyze_quantity_distribution_and_outliers(dataframe):
    print(f"\nMissing values: {dataframe[QUANTITY_COLUMN].isna().sum()}")
    print(f"Unique values: {dataframe[QUANTITY_COLUMN].nunique()}")

    # Outlier detection using IQR method
    Q1 = dataframe[QUANTITY_COLUMN].quantile(0.25)
    Q3 = dataframe[QUANTITY_COLUMN].quantile(0.75)
    IQR = Q3 - Q1

    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR

    outliers = dataframe[(dataframe[QUANTITY_COLUMN] < lower_fence) | (dataframe[QUANTITY_COLUMN] > upper_fence)]
    outlier_count = len(outliers)
    outlier_percentage = (outlier_count / len(dataframe)) * 100

    print(f"Q1 (25th percentile): {Q1}")
    print(f"Q3 (75th percentile): {Q3}")
    print(f"IQR: {IQR}")
    print(f"Lower fence: {lower_fence}")
    print(f"Upper fence: {upper_fence}")
    print(f"\nNumber of outliers: {outlier_count} ({outlier_percentage:.2f}%)")

    if outlier_count > 0:
        print(f"Outlier range: {outliers[QUANTITY_COLUMN].min()} to {outliers[QUANTITY_COLUMN].max()}")

    return Q1, Q3, IQR, outlier_count, outliers


# Apply Min-Max Normalization to Quantity attribute
# Formula: (X - X_min) / (X_max - X_min)
# Returns dataframe with new 'Quantity_Normalized' column in range [0, 1]
def apply_min_max_normalization_to_quantity(dataframe):
    print("MIN-MAX NORMALIZATION")
    min_max_scaler = MinMaxScaler()
    dataframe['Quantity_Normalized'] = min_max_scaler.fit_transform(dataframe[[QUANTITY_COLUMN]])

    print(f"\nOriginal range: [{dataframe[QUANTITY_COLUMN].min()}, {dataframe[QUANTITY_COLUMN].max()}]")
    print(f"Scaled range: [{dataframe['Quantity_Normalized'].min():.6f}, {dataframe['Quantity_Normalized'].max():.6f}]")
    print(f"\nStatistics:")
    print(dataframe['Quantity_Normalized'].describe())

    return dataframe


# Apply Z-Score Standardization to Quantity attribute
# Formula: (X - �) / �
# Returns dataframe with new 'Quantity_Standardized' column (mean=0, std=1)
def apply_z_score_standardization_to_quantity(dataframe):
    print("Z-SCORE STANDARDIZATION")
    standard_scaler = StandardScaler()
    dataframe['Quantity_Standardized'] = standard_scaler.fit_transform(dataframe[[QUANTITY_COLUMN]])

    print(f"\nOriginal mean: {dataframe[QUANTITY_COLUMN].mean():.6f}")
    print(f"Original std: {dataframe[QUANTITY_COLUMN].std():.6f}")
    print(f"\nScaled mean: {dataframe['Quantity_Standardized'].mean():.6f}")
    print(f"Scaled std: {dataframe['Quantity_Standardized'].std():.6f}")
    print(f"\nStatistics:")
    print(dataframe['Quantity_Standardized'].describe())

    return dataframe


# Apply Robust Scaling to Quantity attribute (RECOMMENDED METHOD)
# Formula: (X - median) / IQR
# Returns dataframe with new 'Quantity_Robust' column (median=0, IQR=1)
def apply_robust_scaling_to_quantity(dataframe):
    print("ROBUST SCALING")
    robust_scaler = RobustScaler()
    dataframe['Quantity_Robust'] = robust_scaler.fit_transform(dataframe[[QUANTITY_COLUMN]])

    print(f"\nOriginal median: {dataframe[QUANTITY_COLUMN].median():.6f}")

    IQR = dataframe[QUANTITY_COLUMN].quantile(0.75) - dataframe[QUANTITY_COLUMN].quantile(0.25)
    print(f"Original IQR: {IQR:.6f}")
    print(f"\nScaled median: {dataframe['Quantity_Robust'].median():.6f}")

    scaled_IQR = dataframe['Quantity_Robust'].quantile(0.75) - dataframe['Quantity_Robust'].quantile(0.25)
    print(f"Scaled IQR: {scaled_IQR:.6f}")
    print(f"\nStatistics:")
    print(dataframe['Quantity_Robust'].describe())

    return dataframe


# Compare all three rescaling methods with statistical summary
# Displays side-by-side comparison table of min, max, mean, median, std for each method
def compare_all_rescaling_methods(dataframe):
    print("COMPARISON OF ALL METHODS")
    comparison_df = pd.DataFrame({
        'Method': ['Original', 'Normalization', 'Standardization', 'Robust Scaling'],
        'Column': [QUANTITY_COLUMN, 'Quantity_Normalized', 'Quantity_Standardized', 'Quantity_Robust'],
        'Min': [
            dataframe[QUANTITY_COLUMN].min(),
            dataframe['Quantity_Normalized'].min(),
            dataframe['Quantity_Standardized'].min(),
            dataframe['Quantity_Robust'].min()
        ],
        'Max': [
            dataframe[QUANTITY_COLUMN].max(),
            dataframe['Quantity_Normalized'].max(),
            dataframe['Quantity_Standardized'].max(),
            dataframe['Quantity_Robust'].max()
        ],
        'Mean': [
            dataframe[QUANTITY_COLUMN].mean(),
            dataframe['Quantity_Normalized'].mean(),
            dataframe['Quantity_Standardized'].mean(),
            dataframe['Quantity_Robust'].mean()
        ],
        'Median': [
            dataframe[QUANTITY_COLUMN].median(),
            dataframe['Quantity_Normalized'].median(),
            dataframe['Quantity_Standardized'].median(),
            dataframe['Quantity_Robust'].median()
        ],
        'Std': [
            dataframe[QUANTITY_COLUMN].std(),
            dataframe['Quantity_Normalized'].std(),
            dataframe['Quantity_Standardized'].std(),
            dataframe['Quantity_Robust'].std()
        ]
    })


    print(comparison_df.to_string(index=False))


# Save all three rescaled datasets to separate CSV files
# Creates output directory if needed and saves norm, std, robust versions
def save_all_rescaled_datasets(dataframe, output_directory):
    # Create output directory if it doesn't exist
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save all three versions
    output_files = {
        'normalization': output_path / 'data_rescaling_norm_quantity.csv',
        'standardization': output_path / 'data_rescaling_std_quantity.csv',
        'robust': output_path / 'data_rescaling_robust_quantity.csv'
    }

    # Save normalized version
    dataframe.to_csv(output_files['normalization'], index=False)

    # Save standardized version
    dataframe.to_csv(output_files['standardization'], index=False)

    # Save robust scaled version (recommended)
    dataframe.to_csv(output_files['robust'], index=False)


# Validate all rescaled columns for correctness and data integrity
# Checks ranges, missing values, row count preservation, and correlation with original
def validate_rescaled_data_quality(dataframe):
    print("VALIDATION CHECKS")

    # 1. Range validation
    print("\n1. RANGE VALIDATION:")
    norm_in_range = (dataframe['Quantity_Normalized'] >= 0).all() and (dataframe['Quantity_Normalized'] <= 1).all()
    print(f"Normalization in [0, 1]: {norm_in_range}")

    std_min = dataframe['Quantity_Standardized'].min()
    std_max = dataframe['Quantity_Standardized'].max()
    print(f"Standardization range: [{std_min:.2f}, {std_max:.2f}]")

    robust_min = dataframe['Quantity_Robust'].min()
    robust_max = dataframe['Quantity_Robust'].max()
    print(f"   Robust scaling range: [{robust_min:.2f}, {robust_max:.2f}]")

    # 2. Missing values check
    print("\n2. MISSING VALUES CHECK:")
    print(f"Normalized missing: {dataframe['Quantity_Normalized'].isna().sum()}")
    print(f"Standardized missing: {dataframe['Quantity_Standardized'].isna().sum()}")
    print(f"Robust missing: {dataframe['Quantity_Robust'].isna().sum()}")

    # 3. Row count preservation
    print("\n3. ROW COUNT PRESERVATION:")
    expected_rows = 11971
    actual_rows = len(dataframe)
    print(f"Expected: {expected_rows} rows")
    print(f"Actual: {actual_rows} rows")
    print(f"Match: {actual_rows == expected_rows}")

    # 4. Correlation with original
    print("\n4. CORRELATION WITH ORIGINAL:")
    corr_norm = dataframe[QUANTITY_COLUMN].corr(dataframe['Quantity_Normalized'])
    corr_std = dataframe[QUANTITY_COLUMN].corr(dataframe['Quantity_Standardized'])
    corr_robust = dataframe[QUANTITY_COLUMN].corr(dataframe['Quantity_Robust'])

    print(f"Normalization correlation: {corr_norm:.6f}")
    print(f"Standardization correlation: {corr_std:.6f}")
    print(f"Robust correlation: {corr_robust:.6f}")



def main():
    working_data = load_cleaned_dataset_for_rescaling(INPUT_CSV)

    analyze_quantity_distribution_and_outliers(working_data)

    working_data = apply_min_max_normalization_to_quantity(working_data)
    working_data = apply_z_score_standardization_to_quantity(working_data)
    working_data = apply_robust_scaling_to_quantity(working_data)

    compare_all_rescaling_methods(working_data)


    save_all_rescaled_datasets(working_data, OUTPUT_DIR)

    validate_rescaled_data_quality(working_data)
    print("Recommendation: Robust Scaling")


if __name__ == "__main__":
    main()
