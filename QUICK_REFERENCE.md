# Missing Data Analysis - Quick Reference Guide

## 📊 At a Glance

**Dataset:** Deliverable1Dataset.csv  
**Total Rows:** 12,575 → 11,971 (95.2% retained)  
**Columns with Missing Data:** 5 out of 11

---

## 🎯 Answer Summary

### A. Missing Data Types

| Column | Missing % | Type | Key Evidence |
|--------|-----------|------|--------------|
| **Discount Applied** | 33.39% | **MCAR** | Uniform distribution, no patterns |
| **Quantity** | 4.80% | **MAR** | Depends on Item field |
| **Total Spent** | 4.80% | **MAR** | Depends on Item field, co-missing with Quantity |
| **Price Per Unit** | 4.84% | **MAR** | Depends on Item field, 100% reconstructable |
| **Item** | 9.65% | **MAR** | Varies by Category (8.23%-10.41%) |

### B. Handling Methods

| Step | Column | Method | Justification | Result |
|------|--------|--------|---------------|--------|
| 1 | **Total Spent** | Delete rows | Critical variable, irrecoverable, only 4.8% | Removed 604 rows |
| 2 | **Price Per Unit** | Formula: Total÷Qty | 100% reconstructable, zero error | Imputed 609 values |
| 3 | **Item** | Mode by Category | Category predicts Item, conservative | Imputed 609 values |
| 4 | **Discount Applied** | "Unknown" category | MCAR, 33% too much to drop | Handled 3,988 values |

---

## 📋 Detailed Classifications

### MCAR (Missing Completely At Random)
**Column:** Discount Applied (33.39%)

✅ **Why MCAR:**
- Uniform across all categories (~33%)
- Balanced TRUE/FALSE distribution in observed data
- No relationship with other variables

### MAR (Missing At Random)  
**Columns:** Item, Price Per Unit, Quantity, Total Spent

✅ **Why MAR:**
- Missingness depends on **observable** variables (Item, Category)
- Systematic patterns across categories
- Co-missingness structure:
  - Quantity ↔ Total Spent: 100% overlap (604 cases)
  - Item → Price: 100% overlap (609 cases)
  - Item → Qty/Total: 49.8% overlap (604 cases)

---

## 🔄 Co-Missingness Patterns

```
Item (1,213 missing) splits into:
├── 604 cases: Item + Quantity + Total Spent all missing
└── 609 cases: Item + Price Per Unit missing (but Qty & Total present)

When Item is missing:
├── Price Per Unit is ALWAYS missing (609/609 = 100%)
├── Quantity is missing 49.8% of time (604/1,213)
└── Total Spent is missing 49.8% of time (604/1,213)

When Quantity is missing:
└── Total Spent is ALWAYS missing (604/604 = 100%)
```

---

## 🎬 Step-by-Step Rationale

### STEP 1: Drop Total Spent (604 rows)
```
Before: 12,575 rows
After:  11,971 rows (95.2% retention)
```

**Why delete, not impute?**
- ✅ Critical target variable for analysis
- ✅ Co-missing with Quantity (perfect overlap)
- ✅ Cannot reconstruct from other fields
- ✅ Only 4.8% data loss
- ✅ Eliminates multiple missing value problems

**Side benefits:**
- Quantity becomes 100% complete
- Removes 604 problematic Item cases

---

### STEP 2: Impute Price Per Unit (609 values)
```
Formula: Price = Total Spent ÷ Quantity
Accuracy: 100% (deterministic)
```

**Why formula, not statistical imputation?**
- ✅ Mathematical relationship exists
- ✅ Zero estimation error
- ✅ All complete cases satisfy formula (0 inconsistencies)
- ✅ Both Total and Quantity guaranteed present after Step 1

---

### STEP 3: Impute Item (609 values)
```
Method: Mode (most frequent) by Category
Example: Food missing → impute Item_14_FOOD
```

**Why mode, not other methods?**
- ✅ All missing Items have valid Category
- ✅ Preserves distribution within categories
- ✅ Conservative (no new item codes)
- ✅ Transparent and explainable
- ✅ Appropriate for MAR data

**Alternatives considered:**
- ❌ Predictive model: Too complex, may overfit
- ❌ Random: Doesn't use Category information
- ✅ **Mode: Best balance of simplicity & accuracy**

---

### STEP 4: Handle Discount Applied (3,988 values)
```
Method: Create "Unknown" category
Result: TRUE | FALSE | Unknown
```

**Why "Unknown", not impute?**
- ✅ MCAR pattern (no predictors available)
- ✅ 33% too much to drop
- ✅ Preserves all other data
- ✅ Transparent (doesn't assume TRUE/FALSE)

**Alternatives considered:**
- ❌ Drop: Lose 33% of data
- ⚠️ Random imputation: Adds uncertainty
- ⚠️ Predict from other features: Complex for MCAR
- ✅ **Unknown: Most transparent & preserves data**

---

## ✅ Validation Results

### Mathematical Consistency
```
Total = Price × Quantity
Verified: 11,971 / 11,971 rows (100%)
Inconsistencies: 0
```

### Data Completeness
```
Item:           100% complete ✓
Price Per Unit: 100% complete ✓
Quantity:       100% complete ✓
Total Spent:    100% complete ✓
```

### Category-Item Relationship
```
Each category maintains 25 unique items ✓
All imputed items belong to correct category ✓
```

---

## 📁 Files Generated

| File | Purpose |
|------|---------|
| `Deliverable1Dataset_CLEANED.csv` | Final cleaned dataset |
| `clean_dataset_pipeline.py` | Complete implementation |
| `DELIVERABLE_ANSWERS.md` | Detailed answers to questions A & B |
| `MISSING_DATA_ANALYSIS_REPORT.md` | Full technical report |
| `visualize_missing_data.py` | Create visualizations |
| `QUICK_REFERENCE.md` | This document |

---

## 🚀 How to Run

```bash
# Run the complete pipeline
python3 clean_dataset_pipeline.py

# Generate visualizations
python3 visualize_missing_data.py
```

---

## 📖 Key Concepts

### MCAR (Missing Completely At Random)
- Missingness is **independent** of all variables
- Probability of missing is **the same** for all observations
- Can be safely deleted or imputed

### MAR (Missing At Random)
- Missingness depends on **observed** data, not unobserved
- Probability of missing varies by **observable characteristics**
- Can be handled with appropriate imputation methods

### MNAR (Missing Not At Random)
- Missingness depends on **unobserved** values themselves
- Most problematic type
- **None found in this dataset** ✓

---

## 🎓 Best Practices Applied

1. ✅ Analyze patterns before handling
2. ✅ Handle dependencies first (Total Spent → enables Price imputation)
3. ✅ Use deterministic methods when available (formula > imputation)
4. ✅ Preserve data when possible (only dropped 4.8%)
5. ✅ Validate results (mathematical consistency checks)
6. ✅ Document all decisions (transparent methodology)

---

## 📊 Final Statistics

| Metric | Value |
|--------|-------|
| Original rows | 12,575 |
| Final rows | 11,971 |
| Retention rate | **95.2%** |
| Rows dropped | 604 (4.8%) |
| Values imputed (Price) | 609 (deterministic) |
| Values imputed (Item) | 609 (mode) |
| Values handled (Discount) | 3,988 (category) |
| Mathematical errors | **0** |
| Missing in critical columns | **0** |

---

**Date:** October 2, 2025  
**Course:** CP610 - Data Analysis  
**Deliverable:** #1 - Missing Data Analysis

