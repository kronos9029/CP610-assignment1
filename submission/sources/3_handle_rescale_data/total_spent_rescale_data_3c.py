import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


SCRIPT_DIR = Path(__file__).parent
INPUT_CSV = SCRIPT_DIR / '../../output/1_handle_missing_data/final_cleaned_dataset.csv'
OUTPUT_DIR = SCRIPT_DIR / '../../output/3_handle_rescale_data'
TOTAL_SPENT_COLUMN = 'Total Spent'


# Load the cleaned dataset from the missing data handling phase
# Returns a copy of the dataframe ready for rescaling operations
def load_cleaned_dataset_for_rescaling(input_csv_path):
    original_dataframe = pd.read_csv(input_csv_path)
    working_dataframe = original_dataframe.copy()


    return working_dataframe


# Perform exploratory data analysis on Total Spent attribute
# Calculates descriptive statistics, detects outliers using IQR method, analyzes distribution skewness
# Returns tuple of (Q1, Q3, IQR, outlier_count, outliers_dataframe, mean, median, skewness)
def analyze_total_spent_distribution_and_outliers(dataframe):
    print("DISTRIBUTION ANALYSIS")
    print(dataframe[TOTAL_SPENT_COLUMN].describe())

    # Distribution analysis
    mean_val = dataframe[TOTAL_SPENT_COLUMN].mean()
    median_val = dataframe[TOTAL_SPENT_COLUMN].median()
    std_val = dataframe[TOTAL_SPENT_COLUMN].std()
    skewness = dataframe[TOTAL_SPENT_COLUMN].skew()

    print(f"Mean: ${mean_val:.2f}")
    print(f"Median: ${median_val:.2f}")
    print(f"Standard Deviation: ${std_val:.2f}")
    print(f"Skewness: {skewness:.4f}")

    if abs(mean_val - median_val) > 0.1 * median_val:
        print(f"\nMean (${mean_val:.2f}) differs significantly from Median (${median_val:.2f})")

    if skewness > 0.5:
        print(f"Right-skewed distribution detected (skewness = {skewness:.4f})")
    elif skewness < -0.5:
        print(f"Left-skewed distribution detected (skewness = {skewness:.4f})")
    else:
        print(f"Approximately symmetric distribution (skewness = {skewness:.4f})")

    # Outlier detection using IQR method
    print("OUTLIER DETECTION (IQR METHOD)")

    Q1 = dataframe[TOTAL_SPENT_COLUMN].quantile(0.25)
    Q3 = dataframe[TOTAL_SPENT_COLUMN].quantile(0.75)
    IQR = Q3 - Q1

    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR

    outliers = dataframe[(dataframe[TOTAL_SPENT_COLUMN] < lower_fence) | (dataframe[TOTAL_SPENT_COLUMN] > upper_fence)]
    outlier_count = len(outliers)
    outlier_percentage = (outlier_count / len(dataframe)) * 100

    print(f"Q1 (25th percentile): ${Q1:.2f}")
    print(f"Q3 (75th percentile): ${Q3:.2f}")
    print(f"IQR: ${IQR:.2f}")
    print(f"Lower Fence: ${lower_fence:.2f}")
    print(f"Upper Fence: ${upper_fence:.2f}")
    print(f"\nOutliers detected: {outlier_count} ({outlier_percentage:.2f}% of data)")

    if outlier_count > 0:
        print(f"\nOutlier value range: ${outliers[TOTAL_SPENT_COLUMN].min():.2f} - ${outliers[TOTAL_SPENT_COLUMN].max():.2f}")
        print(f"95th percentile: ${dataframe[TOTAL_SPENT_COLUMN].quantile(0.95):.2f}")
        print(f"99th percentile: ${dataframe[TOTAL_SPENT_COLUMN].quantile(0.99):.2f}")
        print(f"Maximum value: ${dataframe[TOTAL_SPENT_COLUMN].max():.2f}")

    return Q1, Q3, IQR, outlier_count, outliers, mean_val, median_val, skewness


# Apply Min-Max Normalization to Total Spent attribute
# Returns dataframe with new 'TotalSpent_Normalized' column in range [0, 1]
def apply_min_max_normalization_to_total_spent(dataframe):
    print("APPLYING MIN-MAX METHOD")
    min_max_scaler = MinMaxScaler()
    dataframe['TotalSpent_Normalized'] = min_max_scaler.fit_transform(dataframe[[TOTAL_SPENT_COLUMN]])

    return dataframe


# Apply Z-Score Standardization to Total Spent attribute
# Returns dataframe with new 'TotalSpent_Standardized' column (mean=0, std=1)
def apply_z_score_standardization_to_total_spent(dataframe):
    print("APPLYING Z-Score METHOD")

    standard_scaler = StandardScaler()
    dataframe['TotalSpent_Standardized'] = standard_scaler.fit_transform(dataframe[[TOTAL_SPENT_COLUMN]])

    return dataframe


# Apply Robust Scaling to Total Spent attribute (RECOMMENDED METHOD)
# Returns dataframe with new 'TotalSpent_Robust' column (median=0, IQR=1)
def apply_robust_scaling_to_total_spent(dataframe):
    print("APPLYING Robust Scaling METHOD")

    robust_scaler = RobustScaler()
    dataframe['TotalSpent_Robust'] = robust_scaler.fit_transform(dataframe[[TOTAL_SPENT_COLUMN]])

    return dataframe


# Compare all three rescaling methods with statistical summary
# Displays side-by-side comparison table of min, max, mean, median, std dev for each method
def compare_all_rescaling_methods(dataframe):
    comparison_df = pd.DataFrame({
        'Method': ['Original', 'Normalization', 'Standardization', 'Robust Scaling'],
        'Column': [TOTAL_SPENT_COLUMN, 'TotalSpent_Normalized', 'TotalSpent_Standardized', 'TotalSpent_Robust'],
        'Min': [
            dataframe[TOTAL_SPENT_COLUMN].min(),
            dataframe['TotalSpent_Normalized'].min(),
            dataframe['TotalSpent_Standardized'].min(),
            dataframe['TotalSpent_Robust'].min()
        ],
        'Max': [
            dataframe[TOTAL_SPENT_COLUMN].max(),
            dataframe['TotalSpent_Normalized'].max(),
            dataframe['TotalSpent_Standardized'].max(),
            dataframe['TotalSpent_Robust'].max()
        ],
        'Mean': [
            dataframe[TOTAL_SPENT_COLUMN].mean(),
            dataframe['TotalSpent_Normalized'].mean(),
            dataframe['TotalSpent_Standardized'].mean(),
            dataframe['TotalSpent_Robust'].mean()
        ],
        'Median': [
            dataframe[TOTAL_SPENT_COLUMN].median(),
            dataframe['TotalSpent_Normalized'].median(),
            dataframe['TotalSpent_Standardized'].median(),
            dataframe['TotalSpent_Robust'].median()
        ],
        'Std Dev': [
            dataframe[TOTAL_SPENT_COLUMN].std(),
            dataframe['TotalSpent_Normalized'].std(),
            dataframe['TotalSpent_Standardized'].std(),
            dataframe['TotalSpent_Robust'].std()
        ]
    })

    print("METHOD COMPARISON")
    print(comparison_df.to_string(index=False))



# Analyze how each method handles outliers in Total Spent
# Shows sample outliers and their scaled values across all three methods
def analyze_outlier_impact(dataframe, outlier_count, outliers):
    print("OUTLIER IMPACT ANALYSIS")

    if outlier_count > 0:
        # Get outlier indices
        outlier_indices = outliers.index

        # Sample outliers for comparison
        sample_size = min(5, outlier_count)
        sample_outliers = outliers.nlargest(sample_size, TOTAL_SPENT_COLUMN)

        print(f"Top {sample_size} Outliers - How Each Method Handles Them:")

        for idx in sample_outliers.index:
            original = dataframe.loc[idx, TOTAL_SPENT_COLUMN]
            normalized = dataframe.loc[idx, 'TotalSpent_Normalized']
            standardized = dataframe.loc[idx, 'TotalSpent_Standardized']
            robust = dataframe.loc[idx, 'TotalSpent_Robust']

            print(f"\nOriginal: ${original:.2f}")
            print(f"Normalized: {normalized:.4f}  (range [0,1])")
            print(f"Standardized: {standardized:.4f}  (unbounded)")
            print(f"Robust: {robust:.4f}  (unbounded, median-centered)")

        # Outlier range comparison
        print("OUTLIER RANGE COMPARISON")

        outlier_comparison = pd.DataFrame({
            'Method': ['Normalization', 'Standardization', 'Robust Scaling'],
            'Outlier Min': [
                dataframe.loc[outlier_indices, 'TotalSpent_Normalized'].min(),
                dataframe.loc[outlier_indices, 'TotalSpent_Standardized'].min(),
                dataframe.loc[outlier_indices, 'TotalSpent_Robust'].min()
            ],
            'Outlier Max': [
                dataframe.loc[outlier_indices, 'TotalSpent_Normalized'].max(),
                dataframe.loc[outlier_indices, 'TotalSpent_Standardized'].max(),
                dataframe.loc[outlier_indices, 'TotalSpent_Robust'].max()
            ],
            'Non-Outlier 95th %ile': [
                dataframe.loc[~dataframe.index.isin(outlier_indices), 'TotalSpent_Normalized'].quantile(0.95),
                dataframe.loc[~dataframe.index.isin(outlier_indices), 'TotalSpent_Standardized'].quantile(0.95),
                dataframe.loc[~dataframe.index.isin(outlier_indices), 'TotalSpent_Robust'].quantile(0.95)
            ]
        })

        print(outlier_comparison.to_string(index=False))

    else:
        print("\nNo outliers detected using IQR method.")



# Save all three rescaled datasets to separate CSV files
# Creates output directory if needed and saves norm, std, robust versions
def save_all_rescaled_datasets(dataframe, output_directory):
    # Create output directory if it doesn't exist
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save all three versions
    output_files = {
        'normalization': output_path / 'data_rescaling_norm_total_spent.csv',
        'standardization': output_path / 'data_rescaling_std_total_spent.csv',
        'robust': output_path / 'data_rescaling_robust_total_spent.csv'
    }

    # Save normalized version
    dataframe.to_csv(output_files['normalization'], index=False)

    # Save standardized version
    dataframe.to_csv(output_files['standardization'], index=False)

    # Save robust scaled version (recommended)
    dataframe.to_csv(output_files['robust'], index=False)


# Validate all rescaled columns for correctness and data integrity
# Checks ranges, missing values, row count preservation, correlation, and mathematical consistency
def validate_rescaled_data_quality(dataframe):
    print("VALIDATION CHECKS")

    # 1. Range validation
    print("\n1. RANGE VALIDATION")
    norm_in_range = (dataframe['TotalSpent_Normalized'] >= 0).all() and (dataframe['TotalSpent_Normalized'] <= 1).all()
    print(f"Normalization in [0, 1]: {norm_in_range}")
    print(f"Actual range: [{dataframe['TotalSpent_Normalized'].min():.6f}, {dataframe['TotalSpent_Normalized'].max():.6f}]")

    std_range = dataframe['TotalSpent_Standardized'].abs().quantile(0.997)
    print(f"Standardization ~99.7% within [-3, 3]: {std_range:.4f}")
    print(f"Actual range: [{dataframe['TotalSpent_Standardized'].min():.4f}, {dataframe['TotalSpent_Standardized'].max():.4f}]")

    robust_iqr = dataframe['TotalSpent_Robust'].quantile(0.75) - dataframe['TotalSpent_Robust'].quantile(0.25)
    print(f"Robust Scaling IQR H 1: {robust_iqr:.6f}")
    print(f"Median: {dataframe['TotalSpent_Robust'].median():.6f}")

    # 2. Missing values check
    print("\n2. MISSING VALUES CHECK")
    print(f"Normalized missing: {dataframe['TotalSpent_Normalized'].isna().sum()}")
    print(f"Standardized missing: {dataframe['TotalSpent_Standardized'].isna().sum()}")
    print(f"Robust missing: {dataframe['TotalSpent_Robust'].isna().sum()}")

    # 3. Correlation with original
    print("\n3. CORRELATION WITH ORIGINAL (Should be 1.0)")
    corr_norm = dataframe[TOTAL_SPENT_COLUMN].corr(dataframe['TotalSpent_Normalized'])
    corr_std = dataframe[TOTAL_SPENT_COLUMN].corr(dataframe['TotalSpent_Standardized'])
    corr_robust = dataframe[TOTAL_SPENT_COLUMN].corr(dataframe['TotalSpent_Robust'])

    print(f"Normalization correlation: {corr_norm:.6f}")
    print(f"Standardization correlation: {corr_std:.6f}")
    print(f"Robust correlation: {corr_robust:.6f}")


def main():
    working_data = load_cleaned_dataset_for_rescaling(INPUT_CSV)

    Q1, Q3, IQR, outlier_count, outliers, mean_val, median_val, skewness = analyze_total_spent_distribution_and_outliers(working_data)

    working_data = apply_min_max_normalization_to_total_spent(working_data)
    working_data = apply_z_score_standardization_to_total_spent(working_data)
    working_data = apply_robust_scaling_to_total_spent(working_data)

    compare_all_rescaling_methods(working_data)

    analyze_outlier_impact(working_data, outlier_count, outliers)


    save_all_rescaled_datasets(working_data, OUTPUT_DIR)

    validate_rescaled_data_quality(working_data)
    print("Recommendation: Robust Scaling")

if __name__ == "__main__":
    main()
