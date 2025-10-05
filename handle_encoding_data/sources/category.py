import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# Configuration
CSV_IN = Path("../output_data/6_item/encoded_item_dataset.csv")
CSV_OUT = Path("../output_data/5_category/encoded_category_dataset.csv")
CATEGORY = "Category"
PREFIX = "cat"  # use 'cat_' prefix to make columns self-explanatory


# Load encoded item dataset from CSV
# Returns dataframe ready for category encoding
def load_encoded_item_dataset(input_csv_path):
    dataframe = pd.read_csv(input_csv_path)
    return dataframe


# Display basic dataset information and check for missing values
# Shows row/column counts and null summary for Category column
def display_dataset_overview(dataframe):
    row_count, col_count = dataframe.shape
    null_summary = dataframe[[CATEGORY]].isnull().sum()

    print(dataframe.head())
    print(f"\nRows: {row_count}, Cols: {col_count}")
    print("\nNulls in Category:")
    print(null_summary)


# Apply one-hot encoding to Category column
# Creates binary 0/1 columns for each category with 'cat_' prefix
# Settings: drop_first=False (keep all categories), dtype=int (clean 0/1 integers)
def apply_one_hot_encoding_to_category(dataframe):
    # One-Hot Encode Category
    category_dummies = pd.get_dummies(
        dataframe[CATEGORY],
        prefix=PREFIX,
        drop_first=False,
        dtype=int
    )

    print("Created dummy columns:", list(category_dummies.columns)[:10], "...")

    # Concatenate with original dataframe (keep original Category)
    dataframe_with_encoding = pd.concat([dataframe, category_dummies], axis=1)

    print(dataframe_with_encoding.head(5))
    print(f"\nNew columns added: {len(category_dummies.columns)}")

    return dataframe_with_encoding, category_dummies


# Perform diagnostic checks to validate one-hot encoding correctness
# Ensures: only 0/1 values, row sums equal 1, displays distribution of categories
def validate_one_hot_encoding_correctness(category_dummies_dataframe):
    # Ensure dummies contain only 0/1
    only_zero_one = category_dummies_dataframe.apply(lambda s: s.isin([0, 1]).all())
    print("All dummy columns are 0/1:", bool(only_zero_one.all()))

    # For any single row, the sum across Category dummies should be 1 (exactly one Category per row)
    row_sums = category_dummies_dataframe.sum(axis=1)
    print("Row sums unique values (should mostly be 1):", row_sums.value_counts().head())

    # Show the distribution of each dummy (first few)
    print("\nDummy column counts (first few):")
    print(category_dummies_dataframe.sum().sort_values(ascending=False).head(10))


# Visualize category frequency distribution with horizontal bar chart
# Shows how many rows fall into each category after one-hot encoding
def visualize_category_distribution(category_dummies_dataframe):
    category_columns = list(category_dummies_dataframe.columns)
    category_counts = category_dummies_dataframe.sum().sort_values()

    plt.figure(figsize=(8, 4))
    sns.barplot(
        x=category_counts.values,
        y=category_counts.index,
        color="steelblue"
    )
    plt.title("Category Frequency after One-Hot Encoding")
    plt.xlabel("Number of Rows")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.show()


# Save the encoded dataset to CSV file
# Creates output directory if needed and saves with 'cat_' prefix columns
def save_encoded_category_dataset(dataframe, output_csv_path):
    # Ensure output directory exists
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(output_csv_path, index=False)

def main():
    # Step 1: Load encoded item dataset
    input_dataframe = load_encoded_item_dataset(CSV_IN)

    # Step 2: Display dataset overview
    display_dataset_overview(input_dataframe)

    # Step 3: Apply one-hot encoding to Category
    encoded_dataframe, category_dummies = apply_one_hot_encoding_to_category(input_dataframe)

    # Step 4: Validate encoding
    validate_one_hot_encoding_correctness(category_dummies)

    # Step 5: Visualize distribution
    visualize_category_distribution(category_dummies)

    # Step 6: Save encoded dataset
    save_encoded_category_dataset(encoded_dataframe, CSV_OUT)


if __name__ == "__main__":
    main()
