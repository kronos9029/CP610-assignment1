# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **data analysis coursework (CP610 Deliverable #1)** focusing on handling missing data, outlier detection, categorical encoding, and feature scaling for a retail transaction dataset.

**Dataset:** `datasource/Deliverable1Dataset.csv` (12,575 rows, 11 columns)
**Final cleaned dataset:** `Handle missing data/output_data/4_discount_applied/final_cleaned_dataset.csv` (11,971 rows)

## Core Workflow

The project follows a **sequential 3-phase pipeline**:

### Phase 1: Missing Data Handling
**Location:** `Handle missing data/`

**Sequential order (MUST follow this sequence):**
1. **Total Spent & Quantity** → Listwise deletion (drop 604 rows with missing Total Spent)
   - Output: `output_data/1_total_spent/total_spent_cleaned.csv`
2. **Price Per Unit** → Deterministic imputation (formula: `Total Spent ÷ Quantity`)
   - Output: `output_data/2_price_per_unit/price_per_unit_reconstructed.csv`
3. **Item** → Mode imputation by Category
   - Output: `output_data/3_item/item_imputed.csv`
4. **Discount Applied** → Create "Unknown" category (MCAR handling)
   - Output: `output_data/4_discount_applied/final_cleaned_dataset.csv`

**Documentation notebooks:** `docs/{attribute_name}/{attribute_name}.ipynb`
**Analysis notebook:** `1_analyse_data.ipynb`

### Phase 2: Categorical Encoding
**Location:** `Handle encoding data/`

**⚠️ CRITICAL:** All encoding steps use the SAME source dataset (`Handle missing data/output_data/4_discount_applied/final_cleaned_dataset.csv`). Do NOT chain outputs sequentially.

**Sequential order (priority-based):**
1. **Customer ID** → Target Encoding (target: Total Spent)
   - Output: `output_data/1_customer_id/encoded_customer_id_dataset.csv`
2. **Item** → Target Encoding (2-fold cross-validation to prevent leakage)
   - Output: `output_data/2_item/encoded_item_dataset.csv`
3. **Category** → One-Hot Encoding (10 categories)
   - Output: `output_data/3_category/encoded_category_dataset.csv`
4. **Location** → Binary Encoding (In-store=0, Online=1)
   - Notebook: `docs/location/location_encode.ipynb`
5. **Payment Method** → One-Hot Encoding (3 categories)
6. **Discount Applied** → One-Hot Encoding (3 categories: True/False/Unknown)

**See:** `CATEGORICAL_ENCODING_STRATEGY.md` for full rationale and methods.

### Phase 3: Numerical Feature Scaling
**Location:** Root directory (`quantity/`, `price_per_unit/`) and `scaled_output_data/`

**Pipeline for each numerical attribute:**
1. Load encoded dataset from Phase 2
2. Detect outliers (IQR method: 1.5×IQR fences)
3. Decide handling strategy (keep/winsorize/transform)
4. Apply Min-Max scaling (chosen method)
5. Save scaled output to `scaled_output_data/{attribute}/`

**Example:** `quantity/quantity_outlier_scaling_pipeline.ipynb`

## Key Files & Documentation

| File | Purpose |
|------|---------|
| `question.md` | Assignment requirements |
| `CATEGORICAL_ENCODING_STRATEGY.md` | Encoding methods, order, and rationale |
| `MISSING_DATA_ANALYSIS_REPORT.md` | Full technical report for missing data |
| `DELIVERABLE_ANSWERS.md` | Answers to assignment questions |
| `QUICK_REFERENCE.md` | Quick lookup guide for missing data |
| `visualize_missing_data.py` | Generate missing data visualizations |

## File Naming Convention

**Requirement from `question.md`:**
- Use **descriptive names** for functions, variables, classes
- Each task in its own file with clear naming (e.g., `rescaling-3-d.py`)
- Output files reflect the task (e.g., `data-rescaling-Norm-3-d.csv`)

**Current structure:**
- Documentation notebooks: `docs/{category}/{task_name}.ipynb`
- Output data: `output_data/{step_number}_{category}/`
- Naming pattern: `{attribute}_{method}_{type}.csv`

## Running Notebooks

**Environment:** Jupyter notebooks (`.ipynb` files)

**To execute:**
```bash
# Install dependencies (if needed)
pip install pandas numpy matplotlib seaborn scipy

# Run notebooks via Jupyter
jupyter notebook

# Or convert to Python and run
jupyter nbconvert --to python notebook.ipynb
python notebook.py
```

**Common imports in notebooks:**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
```

## Data Flow Architecture

```
datasource/Deliverable1Dataset.csv (12,575 rows)
    ↓
Handle missing data/
    ↓ Step 1: Drop missing Total Spent (→ 11,971 rows)
    ↓ Step 2: Reconstruct Price Per Unit
    ↓ Step 3: Impute Item by Category
    ↓ Step 4: Handle Discount Applied → "Unknown"
    ↓
output_data/4_discount_applied/final_cleaned_dataset.csv
    ↓
Handle encoding data/
    ↓ Step 1: Target encode Customer ID
    ↓ Step 2: Target encode Item (2-fold CV)
    ↓ Step 3: One-hot encode Category
    ↓ (Steps 4-6 for Location, Payment, Discount)
    ↓
output_data/3_category/encoded_category_dataset.csv
    ↓
Scaling pipeline (quantity/, price_per_unit/)
    ↓ Outlier detection (IQR)
    ↓ Min-Max scaling
    ↓
scaled_output_data/{attribute}/scaled_{attribute}_dataset.csv
```

## Important Constraints

1. **Order matters:** Missing data must be handled before encoding; encoding before scaling
2. **No chaining in Phase 2:** Each encoding step starts from `final_cleaned_dataset.csv`
3. **Mathematical consistency:** `Total Spent = Price Per Unit × Quantity` must hold for all rows
4. **Data retention:** Final dataset retains 95.2% of original (11,971/12,575 rows)
5. **No Transaction ID encoding:** Excluded from analysis (identifier only)

## Missing Data Classifications (Reference)

| Column | Type | Rationale |
|--------|------|-----------|
| Discount Applied | MCAR | Uniform 33% across all variables |
| Quantity | MAR | Depends on Item field |
| Total Spent | MAR | Depends on Item field, co-missing with Quantity |
| Price Per Unit | MAR | Depends on Item field, 100% reconstructable |
| Item | MAR | Varies by Category (8.23%-10.41%) |

## Encoding Strategy Summary

| Attribute | Method | Reason |
|-----------|--------|--------|
| Customer ID | Target Encoding | 25 unique values; captures spending patterns |
| Item | Target Encoding (2-fold) | 250 unique values; prevents dimensionality explosion |
| Category | One-Hot | 10 categories; manageable, no ordering |
| Location | Binary | 2 values (In-store/Online) |
| Payment Method | One-Hot | 3 categories (Cash/Credit/Digital) |
| Discount Applied | One-Hot | 3 categories (True/False/Unknown) |

**Dimensionality:** 23 columns (vs. 291 if all one-hot) — 92% reduction

## Validation Checklist

After any data transformation:
```python
# 1. No missing values in critical columns
assert df[["Item", "Price Per Unit", "Quantity", "Total Spent"]].isna().sum().sum() == 0

# 2. Mathematical consistency
assert (abs(df["Total Spent"] - df["Price Per Unit"] * df["Quantity"]) < 0.01).all()

# 3. Row count preservation (after Step 1)
assert len(df) == 11971

# 4. One-hot columns sum to 1 per row
assert (df[["Payment_Cash", "Payment_Credit Card", "Payment_Digital Wallet"]].sum(axis=1) == 1).all()
```

## Git Repository Notes

**Current branch:** `master`
**Recent commits focus on:** Scaled pipeline for Quantity/Price Per Unit, encoding documentation

**Unstaged changes include:**
- Moved encoding notebooks to `Handle encoding data/docs/`
- Moved output files to proper directories
- Updated documentation files

**Untracked files:**
- `.idea/` (IDE settings)
- `CATEGORICAL_ENCODING_STRATEGY.md`
- `Handle encoding data/docs/location/`
- `Handle missing data/docs/`