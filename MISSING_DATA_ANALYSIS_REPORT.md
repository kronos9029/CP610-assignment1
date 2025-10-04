# Missing Data Analysis Report

## Dataset Overview
- **Total Rows:** 12,575
- **Columns with Missing Values:** 5 out of 11 columns

## 1. Missing Data Summary

| Column | Missing Count | Missing % | Type of Missing Data |
|--------|---------------|-----------|---------------------|
| **Discount Applied** | 4,199 | 33.39% | MCAR |
| **Item** | 1,213 | 9.65% | MAR |
| **Price Per Unit** | 609 | 4.84% | MAR |
| **Quantity** | 604 | 4.80% | MAR |
| **Total Spent** | 604 | 4.80% | MAR |

---

## 2. Missing Data Type Classification & Rationale

### a) **Discount Applied (33.39% missing) - MCAR (Missing Completely At Random)**

**Classification:** MCAR

**Rationale:**
- Missing values are evenly distributed (~33%) across all categories, payment methods, and locations
- The distribution of TRUE/FALSE values is nearly identical (4,219 vs 4,157)
- The missing pattern shows no relationship with other variables
- This appears to be a data collection issue where the field was simply not recorded for 1/3 of transactions
- The missingness does NOT depend on observed or unobserved data

**Evidence:**
```
TRUE:  4,219 (33.5%)
FALSE: 4,157 (33.1%)
NaN:   4,199 (33.4%)
```

---

### b) **Quantity & Total Spent (4.80% missing) - MAR (Missing At Random)**

**Classification:** MAR

**Rationale:**
- **Perfect co-missingness:** When Quantity is missing, Total Spent is ALWAYS missing (604 occurrences)
- When Item is missing, Quantity and Total Spent are missing in 604 out of 1,213 cases (~49.8%)
- Missing rates vary slightly by category (4.15%-5.69%), suggesting missingness depends on observable characteristics
- The missingness pattern is **systematic and related to the Item field**

**Key Evidence:**
- Quantity missing + Total Spent missing: **604 (100% overlap)**
- Item missing + Quantity missing: **604**
- Item missing + Total Spent missing: **604**
- This suggests that when certain transaction details (Item) were not captured, the associated numeric fields (Quantity, Total Spent) were also not recorded

**Why MAR and not MNAR:**
- The missingness depends on the Item field (an observed variable)
- Not completely random as certain categories show higher rates
- Not MNAR because the missingness is explained by the Item field status, not by the values themselves

---

### c) **Price Per Unit (4.84% missing) - MAR (Missing At Random)**

**Classification:** MAR

**Rationale:**
- **Perfect overlap with Item missingness:** All 609 missing Price Per Unit values occur when Item is also missing
- Co-missingness pattern: Item + Price Per Unit = 609 (all missing Price values)
- Missing rates vary by category (4.09%-5.56%), with Milk Products showing the highest rate
- **100% reconstructable** from Total Spent ÷ Quantity (when both are present)

**Key Evidence:**
- Item missing + Price Per Unit missing: **609 out of 609 (100%)**
- Price Per Unit can be reconstructed for all missing values using: `Price = Total Spent / Quantity`
- The missingness is conditional on Item being missing (observable variable)

**Why MAR:**
- Missingness depends on the Item field (observable)
- The missing values can be explained by the Item recording issue
- Mathematical relationship exists that allows reconstruction

---

### d) **Item (9.65% missing) - MAR (Missing At Random)**

**Classification:** MAR

**Rationale:**
- Missing rates vary systematically by category (8.23%-10.41%)
- Higher missingness in certain categories: Patisserie (10.41%), Computers and electric accessories (10.33%)
- Lower missingness in Furniture (8.23%) and Beverages (8.93%)
- All 1,213 missing Items have valid Category information (100%)
- **The Item field drives missingness in other fields:**
  - When Item is missing, Price Per Unit is ALWAYS missing (609 cases)
  - When Item is missing, Quantity & Total Spent are missing 49.8% of the time (604 cases)

**Key Evidence:**
- Category information is always present, allowing category-based imputation
- Missingness varies by product category, indicating it depends on observable characteristics
- The pattern suggests certain categories had less rigorous item-level data entry

**Why MAR:**
- Missingness depends on Category (observable variable)
- Systematic variation across categories
- Not MCAR because the missing rate is not uniform
- Not MNAR because we can explain the pattern through Category

---

## 3. Co-Missingness Patterns

### Critical Findings:

1. **Quantity ↔ Total Spent:** Perfect overlap (604 cases)
   - When one is missing, the other is ALWAYS missing
   
2. **Item → All Other Fields:**
   - Item missing → Price Per Unit ALWAYS missing (609 cases)
   - Item missing → Quantity & Total Spent missing in 604 cases
   
3. **No overlap between:**
   - Price Per Unit + Quantity: 0
   - Price Per Unit + Total Spent: 0
   
4. **Mathematical Consistency:**
   - All complete rows satisfy: Total Spent = Price Per Unit × Quantity
   - Zero inconsistent calculations (out of 11,362 complete rows)

---

## 4. Recommended Handling Order & Methods

### **Priority Order (Handle in this sequence):**

#### **STEP 1: Drop rows with missing Total Spent (604 rows = 4.8%)**

**Justification:**
- Total Spent is a **critical target variable** for transaction analysis
- These 604 rows also have missing Quantity (perfect overlap)
- Cannot reliably reconstruct: Missing both Quantity AND at least one of (Price or Item)
- Small data loss (4.8%) vs. large gain in data integrity
- After removal, all remaining rows have complete Total Spent & Quantity

**Method:** Listwise deletion (drop rows)

**Impact:**
- Remaining dataset: 11,971 rows (95.2% retention)
- Removes 604 Item missing values
- Keeps 609 Item missing values that CAN be imputed

---

#### **STEP 2: Impute Price Per Unit (609 rows after Step 1)**

**Justification:**
- **100% reconstructable** using mathematical relationship: `Price = Total Spent ÷ Quantity`
- Deterministic imputation (no estimation error)
- Both Total Spent and Quantity are guaranteed present after Step 1
- Maintains mathematical consistency

**Method:** Deterministic imputation using formula

```python
df.loc[df['Price Per Unit'].isna(), 'Price Per Unit'] = \
    df.loc[df['Price Per Unit'].isna(), 'Total Spent'] / \
    df.loc[df['Price Per Unit'].isna(), 'Quantity']
```

**Impact:**
- All Price Per Unit values complete
- Zero estimation error
- Maintains data integrity

---

#### **STEP 3: Impute Item (609 rows after Steps 1-2)**

**Quantify missing Item**
- 609 of 11,971 rows (5.09%) still lack `Item` labels after Steps 1-2.
- All affected rows retain complete numeric features (`Price Per Unit`, `Quantity`, `Total Spent`).

**Missingness mechanism (MAR)**
- Category-level missingness ranges from 5.82% in Milk Products to 4.26% in Furniture, reflecting observable, category-specific data entry gaps.
- Payment Method influences the share of missing items (Digital Wallet 5.71%, Credit Card 5.04%, Cash 4.53%).
- Online orders show a slightly higher missing rate (5.32%) than in-store transactions (4.84%).

**Co-missingness diagnostics**
- Missing `Item` values no longer coincide with numeric fields: overlap with `Price Per Unit`, `Quantity`, and `Total Spent` is zero.
- 205 rows (33.7% of missing items) also lack `Discount Applied`, reinforcing that the Item field drives other missingness.
- Every missing `Item` retains a valid `Category`; per-category counts remain balanced (e.g., Milk Products 88, Food 81, Furniture 65).

**Imputability assessment**
- Each category has at least one observed item, enabling deterministic filling via category-specific modes (e.g., Milk Products → `Item_16_MILK`, Furniture → `Item_25_FUR`).
- Category modes cover 100% of missing rows; a global mode fallback remains available but unnecessary.

**Handling strategy: Category-aware mode imputation**
- Compute the most frequent item per category from observed rows.
- Replace missing items by mapping each Category to its mode; fallback to overall mode for any residual gaps.

```python
import pandas as pd

df = pd.read_csv("output_data/2_price_per_unit/price_per_unit_reconstructed.csv")

cat_mode = (
    df.dropna(subset=["Item", "Category"])
      .groupby("Category")["Item"]
      .agg(lambda s: s.mode().iloc[0])
)

mask = df["Item"].isna() & df["Category"].notna()
df.loc[mask, "Item"] = df.loc[mask, "Category"].map(cat_mode)

if df["Item"].isna().any():
    df["Item"] = df["Item"].fillna(df["Item"].mode().iloc[0])

df.to_csv("Deliverable1Dataset_item_imputed.csv", index=False)
```

**Impact & validation**
- 609 missing `Item` values imputed; `Item` column becomes 100% complete.
- Category-to-item relationships preserved because fills respect observed category modes.
- End-to-end checks confirm no remaining missing values in `Item`, `Price Per Unit`, `Quantity`, or `Total Spent` prior to tackling `Discount Applied`.

**Summary**
- Missingness is MAR, driven by observable category patterns.
- Category-aware mode imputation provides a fast, low-assumption remedy with zero residual gaps.
- Output dataset: `Deliverable1Dataset_item_imputed.csv`, ready for Step 4 handling of `Discount Applied`.

---

#### **STEP 4: Handle Discount Applied (4,199 rows = 33.39%)**

**Justification:**
- MCAR pattern (completely random)
- Binary categorical variable (TRUE/FALSE)
- Large proportion missing (33%)
- Cannot drop due to excessive data loss

**Method:** Multiple options:

**Option A (Conservative):** Create a third category "Unknown"
```python
df['Discount Applied'] = df['Discount Applied'].fillna('Unknown')
```

**Option B (Statistical):** Random imputation based on observed distribution
```python
# Maintain 50/50 TRUE/FALSE ratio observed in complete data
import numpy as np
mask = df['Discount Applied'].isna()
df.loc[mask, 'Discount Applied'] = \
    np.random.choice([True, False], size=mask.sum(), p=[0.5, 0.5])
```

**Option C (Predictive):** Use transaction characteristics
- Check if Total Spent correlates with discount status
- Use logistic regression if patterns exist

**Recommended:** Option A (Unknown category) - safest for MCAR data

---

## 5. Summary of Approach

### Handling Strategy Matrix:

| Column | Missing Type | Method | Justification | Order |
|--------|-------------|---------|---------------|-------|
| **Total Spent** | MAR | Listwise Deletion | Critical variable, co-missing with Quantity | 1 |
| **Quantity** | MAR | (Removed in Step 1) | Perfect overlap with Total Spent | 1 |
| **Price Per Unit** | MAR | Deterministic Imputation | 100% reconstructable via formula | 2 |
| **Item** | MAR | Mode by Category | Category provides strong signal | 3 |
| **Discount Applied** | MCAR | Create "Unknown" category | Large missing %, MCAR pattern | 4 |

### Key Principles Applied:
1. ✅ **Remove irrecoverable data first** (Total Spent + Quantity)
2. ✅ **Reconstruct deterministic values** (Price Per Unit)
3. ✅ **Impute based on strong predictors** (Item via Category)
4. ✅ **Handle MCAR data last** (Discount Applied)
5. ✅ **Minimize assumptions** (use mathematical relationships when possible)
6. ✅ **Preserve data integrity** (maintain consistency checks)

---

## 6. Implementation Code

### Complete Pipeline:

```python
import pandas as pd

# Load data
df = pd.read_csv("datasource/Deliverable1Dataset.csv")

# Convert numeric columns
for col in ["Price Per Unit", "Quantity", "Total Spent"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# STEP 1: Drop rows with missing Total Spent
print(f"Before: {len(df)} rows")
df = df.dropna(subset=["Total Spent"])
print(f"After dropping missing Total Spent: {len(df)} rows")

# STEP 2: Impute Price Per Unit using formula
mask = df["Price Per Unit"].isna()
df.loc[mask, "Price Per Unit"] = df.loc[mask, "Total Spent"] / df.loc[mask, "Quantity"]
print(f"Imputed Price Per Unit: {mask.sum()} values")

# STEP 3: Impute Item using mode by Category
for category in df["Category"].unique():
    mask = (df["Category"] == category) & (df["Item"].isna())
    if mask.sum() > 0:
        mode_item = df[df["Category"] == category]["Item"].mode()
        if len(mode_item) > 0:
            df.loc[mask, "Item"] = mode_item[0]
print(f"Imputed Item values")

# STEP 4: Handle Discount Applied
df["Discount Applied"] = df["Discount Applied"].fillna("Unknown")
print(f"Handled Discount Applied missing values")

# Verify no missing values in critical columns
print("\nFinal Missing Value Count:")
print(df[["Item", "Price Per Unit", "Quantity", "Total Spent"]].isna().sum())

# Save cleaned dataset
df.to_csv("Deliverable1Dataset_CLEANED.csv", index=False)
print("\nSaved cleaned dataset")
```

---

## 7. Expected Results

After applying this pipeline:
- **Rows:** 11,971 (95.2% retention)
- **Item:** 0 missing (100% imputed)
- **Price Per Unit:** 0 missing (100% reconstructed)
- **Quantity:** 0 missing (100% complete)
- **Total Spent:** 0 missing (100% complete)
- **Discount Applied:** 0 missing (handled as "Unknown")

### Data Integrity:
✅ Mathematical consistency maintained (Total = Price × Quantity)
✅ Category-Item relationships preserved
✅ Minimal estimation error (only Item field imputed)
✅ No information loss for critical variables

---

## 8. Validation Checks

After imputation, verify:

```python
# 1. Mathematical consistency
df["Calculated_Total"] = df["Price Per Unit"] * df["Quantity"]
assert (abs(df["Total Spent"] - df["Calculated_Total"]) < 0.01).all()

# 2. No missing values in critical columns
assert df[["Item", "Price Per Unit", "Quantity", "Total Spent"]].isna().sum().sum() == 0

# 3. Category-Item consistency
for category in df["Category"].unique():
    items = df[df["Category"] == category]["Item"].unique()
    print(f"{category}: {len(items)} unique items")
```

---

**Report Generated:** Based on comprehensive analysis of Deliverable1Dataset.csv

