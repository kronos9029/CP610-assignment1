import pandas as pd
import numpy as np

def load_data(file_path):
    source = pd.read_csv(file_path)
    data = source.copy()
    return data

def handle_item_missing(data):
    # Find all the observations that missing values
    total_missing_items = data['Item'].isnull().sum()
    print(f'Total number of missing rows for Item: {total_missing_items}')

    total_missing_items_observations = data[data['Item'].isnull()]
    total_missing_items_observations.head(50)

    # Evaluate where `Item` values are absent to understand dependencies.
    item_missing = data['Item'].isna()
    print(f'Missing Item count: {item_missing.sum()} of {len(data)} rows ({item_missing.mean():.2%})')
    print('Share of Item missing by Category (top 5):')
    print(data.assign(item_missing=item_missing).groupby('Category')['item_missing'].mean().sort_values(ascending=False).head())
    print('Share of Item missing by Payment Method:')
    print(data.assign(item_missing=item_missing).groupby('Payment Method')['item_missing'].mean().sort_values(ascending=False))

    # Check for Category+Price combinations without unique Item mapping
    lookup_conflicts = (
        data.dropna(subset=['Category', 'Price Per Unit', 'Item'])
        .groupby(['Category', 'Price Per Unit'])['Item']
        .nunique()
    )
    conflicts = lookup_conflicts[lookup_conflicts > 1]
    print(f'Category+Price combinations without unique Item mapping: {len(conflicts)}')

    # Fill in missing values for Item
    # Flag rows where Item and Price per Unit are missing but quantity and total are available
    needs_price = (
        data['Item'].isna() &
        data['Price Per Unit'].isna() &
        data['Quantity'].notna() &
        data['Total Spent'].notna()
    )
    # Recompute unit prices for those rows before attempting Item lookup
    data.loc[needs_price, 'Price Per Unit'] = (
        data.loc[needs_price, 'Total Spent'] / data.loc[needs_price, 'Quantity']
    ).round(1)
    
    # Build Category+Price lookup table mapping back to the expected Item code
    # only need to set index for all the rows that have the values of the category and price per unit (remove all the rows that have invalid values for outliers)
    # do the same with duplicated rows then create a mapping table between [category and price per unit] and item e.g ('Furniture', 6.5) -> Item_18_FOOD
    item_lookup = (
        data.dropna(subset=['Category', 'Price Per Unit', 'Item'])
        .drop_duplicates(subset=['Category', 'Price Per Unit'])
        .set_index(['Category', 'Price Per Unit'])['Item']
    )

    print(item_lookup)
    # Apply lookup to rows still missing Item values
    # create a mapping table between category and price per unit e.g, ('Food', 11.0), ('Furniture', 6.5)
    keys = list(zip(data['Category'], data['Price Per Unit']))
    data['Item'] = data['Item'].fillna(pd.Series(keys, index=data.index).map(item_lookup))

    return data

def save_data(data, file_path):
    data.to_csv(file_path, index=False)

def main():
    data = load_data("../Deliverable1Dataset.csv")
    data = handle_item_missing(data)
    save_data(data, "dataset_with_item_imputed.csv")

if __name__ == "__main__":
    main()