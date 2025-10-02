"""
Visual Summary of Missing Data Analysis
Creates before/after comparison visualization
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load original and cleaned datasets
df_original = pd.read_csv("Deliverable1Dataset.csv")
df_cleaned = pd.read_csv("Deliverable1Dataset_CLEANED.csv")

# Convert numeric columns
for col in ["Price Per Unit", "Quantity", "Total Spent"]:
    df_original[col] = pd.to_numeric(df_original[col], errors="coerce")

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Missing Data Analysis - Before & After', fontsize=16, fontweight='bold')

# ============================================================================
# Plot 1: Overall Missing Percentages
# ============================================================================
ax1 = axes[0, 0]

columns_with_missing = ["Item", "Price Per Unit", "Quantity", "Total Spent", "Discount Applied"]
missing_before = [(df_original[col].isna().sum() / len(df_original) * 100) for col in columns_with_missing]
missing_after = [(df_cleaned[col].isna().sum() / len(df_cleaned) * 100) for col in columns_with_missing]

x = np.arange(len(columns_with_missing))
width = 0.35

bars1 = ax1.bar(x - width/2, missing_before, width, label='Before', color='#e74c3c', alpha=0.8)
bars2 = ax1.bar(x + width/2, missing_after, width, label='After', color='#27ae60', alpha=0.8)

ax1.set_xlabel('Column', fontweight='bold')
ax1.set_ylabel('Missing %', fontweight='bold')
ax1.set_title('Missing Data Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(columns_with_missing, rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=8)

# ============================================================================
# Plot 2: Missing Data by Category (Original)
# ============================================================================
ax2 = axes[0, 1]

categories = df_original['Category'].unique()
missing_by_cat = []
for cat in categories:
    cat_data = df_original[df_original['Category'] == cat]
    total_cells = len(cat_data) * 5  # 5 columns with potential missing values
    missing_cells = cat_data[columns_with_missing].isna().sum().sum()
    missing_pct = (missing_cells / total_cells * 100)
    missing_by_cat.append(missing_pct)

colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
bars = ax2.barh(range(len(categories)), missing_by_cat, color=colors)
ax2.set_yticks(range(len(categories)))
ax2.set_yticklabels(categories, fontsize=9)
ax2.set_xlabel('Missing %', fontweight='bold')
ax2.set_title('Missing Data by Category (Original)')
ax2.grid(axis='x', alpha=0.3)

for i, bar in enumerate(bars):
    width = bar.get_width()
    ax2.text(width, bar.get_y() + bar.get_height()/2.,
            f'{width:.1f}%',
            ha='left', va='center', fontsize=8, fontweight='bold')

# ============================================================================
# Plot 3: Data Retention
# ============================================================================
ax3 = axes[1, 0]

retention_data = [
    len(df_original),
    len(df_cleaned),
    len(df_original) - len(df_cleaned)
]
retention_labels = [
    f'Original\n{retention_data[0]:,} rows',
    f'Retained\n{retention_data[1]:,} rows\n(95.2%)',
    f'Dropped\n{retention_data[2]:,} rows\n(4.8%)'
]
colors_ret = ['#3498db', '#27ae60', '#e74c3c']

bars = ax3.bar(range(3), retention_data, color=colors_ret, alpha=0.8)
ax3.set_xticks(range(3))
ax3.set_xticklabels(retention_labels)
ax3.set_ylabel('Row Count', fontweight='bold')
ax3.set_title('Data Retention Summary')
ax3.grid(axis='y', alpha=0.3)

# ============================================================================
# Plot 4: Missing Data Classification
# ============================================================================
ax4 = axes[1, 1]

# Classification summary
classifications = {
    'MCAR\n(Discount Applied)': {'count': 1, 'color': '#3498db'},
    'MAR\n(Item, Price, Qty, Total)': {'count': 4, 'color': '#e67e22'}
}

labels = list(classifications.keys())
counts = [classifications[k]['count'] for k in labels]
colors_class = [classifications[k]['color'] for k in labels]

wedges, texts, autotexts = ax4.pie(counts, labels=labels, colors=colors_class,
                                     autopct='%1.0f%%', startangle=90,
                                     textprops={'fontsize': 10, 'fontweight': 'bold'})
ax4.set_title('Missing Data Classification')

# Add legend with handling methods
legend_labels = [
    'MCAR → "Unknown" category',
    'MAR → Delete/Impute'
]
ax4.legend(legend_labels, loc='upper left', bbox_to_anchor=(0.8, 0, 0.5, 1), fontsize=9)

plt.tight_layout()
plt.savefig('missing_data_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved as 'missing_data_analysis.png'")
plt.show()

# ============================================================================
# Text Summary
# ============================================================================
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"\nOriginal Dataset:")
print(f"  Rows: {len(df_original):,}")
print(f"  Missing values: {df_original[columns_with_missing].isna().sum().sum():,}")
print(f"  Missing percentage: {(df_original[columns_with_missing].isna().sum().sum() / (len(df_original) * 5) * 100):.2f}%")

print(f"\nCleaned Dataset:")
print(f"  Rows: {len(df_cleaned):,}")
print(f"  Missing values (critical columns): {df_cleaned[['Item', 'Price Per Unit', 'Quantity', 'Total Spent']].isna().sum().sum()}")
print(f"  Data retention: {(len(df_cleaned) / len(df_original) * 100):.2f}%")

print(f"\nHandling Summary:")
print(f"  ✓ Dropped {len(df_original) - len(df_cleaned):,} rows (4.8%) - irrecoverable Total Spent")
print(f"  ✓ Imputed 609 Price Per Unit values - deterministic formula")
print(f"  ✓ Imputed 609 Item values - mode by category")
print(f"  ✓ Handled 3,988 Discount Applied - 'Unknown' category")

print("\n" + "="*80)

