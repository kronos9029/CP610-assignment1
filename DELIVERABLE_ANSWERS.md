# CP610 Deliverable #1 - Missing Data Analysis

## Question A: Identify the Type of Missing Data (MCAR/MAR/MNAR)

### Summary Table

| Column | Missing % | Classification | Rationale |
|--------|-----------|----------------|-----------|
| **Discount Applied** | 33.39% | **MCAR** | Completely random distribution across all variables |
| **Quantity** | 4.80% | **MAR** | Depends on Item field (observable variable) |
| **Total Spent** | 4.80% | **MAR** | Depends on Item field (observable variable) |
| **Price Per Unit** | 4.84% | **MAR** | Depends on Item field (observable variable) |
| **Item** | 9.65% | **MAR** | Varies systematically by Category (observable) |

---

## Detailed Classification & Rationale

### 1. **Discount Applied - MCAR (Missing Completely At Random)**

**Evidence:**
- Missing rate is uniform: ~33% across ALL categories, payment methods, and locations
- Distribution of observed values is balanced: TRUE (33.5%) vs FALSE (33.1%) vs Missing (33.4%)
- No relationship with any other variable in the dataset
- Appears to be a data collection/recording issue

**Why MCAR:**
- The probability of missingness does NOT depend on observed OR unobserved data
- Random data entry issue where field was simply not filled
- No systematic pattern detected

---

### 2. **Quantity & Total Spent - MAR (Missing At Random)**

**Evidence:**
- **Perfect co-missingness:** These two fields are ALWAYS missing together (604 cases)
- Missing rate varies by category: 4.15% (Furniture) to 5.69% (Patisserie)
- 100% of missing Quantity/Total Spent cases also have missing Item (604 out of 604)
- When Item is missing, Quantity/Total Spent are missing 49.8% of the time

**Why MAR (not MNAR or MCAR):**
- Missingness depends on the Item field status (an **observable** variable) → Not MNAR
- Missing rates vary by category, showing systematic patterns → Not MCAR
- The missingness can be explained by the Item recording issue
- No evidence that the missing values depend on their own unobserved values

**Systematic Pattern:**
```
Item missing (1,213) splits into:
  → 604 cases: Item + Quantity + Total Spent all missing
  → 609 cases: Only Item + Price Per Unit missing
```

---

### 3. **Price Per Unit - MAR (Missing At Random)**

**Evidence:**
- ALL 609 missing Price Per Unit values occur when Item is also missing
- Missing rate varies by category: 4.09% (Furniture) to 5.56% (Milk Products)
- **100% reconstructable** from existing data: Price = Total Spent ÷ Quantity
- Mathematical relationship holds for all complete cases (0 inconsistencies out of 11,362 rows)

**Why MAR:**
- Missingness is conditional on Item being missing (an **observable** variable)
- The missing values can be completely explained by the Item field status
- Systematic variation across categories rules out MCAR
- Not MNAR because missingness doesn't depend on the unobserved price values themselves

---

### 4. **Item - MAR (Missing At Random)**

**Evidence:**
- Missing rate varies systematically by category:
  - Highest: Patisserie (10.41%), Computers (10.33%)
  - Lowest: Furniture (8.23%), Beverages (8.93%)
- ALL 1,213 missing Items have valid Category information (100%)
- Item missingness drives missingness in other fields:
  - Item missing → Price ALWAYS missing (609/1,213 = 50.2%)
  - Item missing → Quantity & Total ALWAYS missing (604/1,213 = 49.8%)

**Why MAR:**
- Missingness depends on Category (an **observable** variable)
- Systematic variation across categories indicates non-random pattern
- Not MCAR: Missing rate is NOT uniform (ranges from 8.23% to 10.41%)
- Not MNAR: We can explain the pattern through Category; no evidence that missingness depends on the unobserved item values themselves

**Interpretation:**
Certain product categories (Patisserie, Computers) had less rigorous item-level data entry compared to others (Furniture, Beverages). This is a data collection quality issue that varies by department/category.

---

## Question B: Handle Missing Data - Method Justification

### Recommended Approach: **Sequential 4-Step Pipeline**

**Order matters!** Handle columns with dependencies first.

---

### **STEP 1: Drop rows with missing Total Spent** ✓ IMPLEMENTED

**Method:** Listwise deletion (remove 604 rows = 4.8%)

**Justification:**
1. **Critical target variable** - Total Spent is essential for transaction analysis
2. **Irrecoverable data** - These rows have missing Quantity AND at least one other key field
3. **Perfect co-missingness** - When Total Spent is missing, Quantity is ALWAYS missing
4. **Small data loss** - Only 4.8% of dataset vs. preserving 95.2% high-quality data
5. **Cannot reliably impute** - No mathematical relationship allows reconstruction
6. **Strategic benefit** - Removing these rows solves multiple missing value problems:
   - Eliminates all Quantity missing values
   - Eliminates 604 out of 1,213 Item missing values

**Result:** 11,971 rows retained (95.2% retention rate)

---

### **STEP 2: Impute Price Per Unit using formula** ✓ IMPLEMENTED

**Method:** Deterministic imputation using mathematical relationship

```python
Price Per Unit = Total Spent ÷ Quantity
```

**Justification:**
1. **100% reconstructable** - All 609 missing prices can be calculated exactly
2. **Zero estimation error** - This is not imputation, it's reconstruction
3. **Maintains mathematical consistency** - The formula Price × Quantity = Total holds for ALL complete cases
4. **No assumptions required** - Using known mathematical relationship
5. **Best practice** - When deterministic relationships exist, use them first

**Validation:** All 11,362 complete rows satisfy the formula with 0 inconsistencies

**Result:** Price Per Unit is now 100% complete with perfect accuracy

---

### **STEP 3: Impute Item using mode by category** ✓ IMPLEMENTED

**Method:** Mode imputation (most frequent value) within each Category

**Justification:**
1. **Strong predictor** - Category provides strong signal for Item (100% of missing Items have valid Category)
2. **Preserves distribution** - Mode imputation maintains the frequency distribution within categories
3. **Conservative approach** - Doesn't introduce new item codes that don't exist in data
4. **MAR appropriate** - Mode imputation is suitable for MAR data when a strong predictor exists
5. **Category consistency** - Ensures imputed Items belong to the correct product category

**Alternative considered:** Predictive imputation using Price Per Unit ranges
- More sophisticated but may overfit
- Mode is simpler and more transparent

**Example Imputations:**
- Food category: 81 missing → imputed with Item_14_FOOD (most common in Food)
- Computers category: 80 missing → imputed with Item_19_CEA (most common in Computers)

**Result:** Item is now 100% complete, all imputed values are category-consistent

---

### **STEP 4: Handle Discount Applied as 'Unknown' category** ✓ IMPLEMENTED

**Method:** Create a third category "Unknown" for missing values

**Justification:**
1. **MCAR pattern** - Missing completely at random, so imputation won't introduce bias
2. **Large proportion** - 33.4% missing is too much to drop (would lose ~4,000 rows)
3. **Preserves information** - Keeping these rows as "Unknown" retains all other valuable data
4. **Transparent** - Analysis can explicitly handle "Unknown" vs. TRUE/FALSE
5. **No false assumptions** - Don't assume TRUE or FALSE when we don't know

**Alternative methods considered:**

| Method | Pros | Cons | Decision |
|--------|------|------|----------|
| Drop rows | Clean data | Lose 33% of data | ❌ Too much loss |
| Random imputation | Maintains distribution | Adds uncertainty | ⚠️ Possible but less transparent |
| Predictive model | Uses other features | Complex, may introduce bias | ⚠️ Overly complex for MCAR |
| **"Unknown" category** | **Transparent, preserves data** | **Adds third category** | ✅ **CHOSEN** |

**Result:** All missing values handled, maintains 95.2% data retention

---

## Why This Order?

```
Priority: Dependencies → Deterministic → Predictive → Random

1. Total Spent (Delete)     ← Solves Quantity too (perfect overlap)
                            ↓
2. Price Per Unit (Formula) ← 100% reconstructable, zero error
                            ↓
3. Item (Mode by Category)  ← Predictive using Category
                            ↓
4. Discount (Unknown)       ← MCAR, no dependencies
```

**Key Principle:** Handle the most constrained/dependent variables first, then work toward less constrained ones.

---

## Final Results

### Data Quality Metrics:

| Metric | Value |
|--------|-------|
| **Original rows** | 12,575 |
| **Final rows** | 11,971 |
| **Retention rate** | 95.2% |
| **Rows dropped** | 604 (4.8%) |
| **Missing values remaining** | 0 in critical columns |
| **Mathematical consistency** | 100% (all rows satisfy Price × Quantity = Total) |

### Missing Value Resolution:

| Column | Before | After | Method |
|--------|--------|-------|--------|
| Total Spent | 604 (4.80%) | 0 (0%) | ✓ Deleted rows |
| Quantity | 604 (4.80%) | 0 (0%) | ✓ Deleted rows |
| Price Per Unit | 609 (4.84%) | 0 (0%) | ✓ Formula imputation |
| Item | 1,213 (9.65%) | 0 (0%) | ✓ Mode by category |
| Discount Applied | 4,199 (33.39%) | 0 (0%) | ✓ "Unknown" category |

---

## Validation Performed

✅ **Mathematical Consistency:** 11,971 rows all satisfy Total = Price × Quantity (0 errors)  
✅ **Category-Item Consistency:** All 8 categories maintain 25 unique items each  
✅ **No Critical Missing Values:** Item, Price, Quantity, Total all 100% complete  
✅ **Data Integrity:** No duplicate Transaction IDs, all dates valid  

---

## Code Implementation

**Complete pipeline available in:** `clean_dataset_pipeline.py`

**Output file:** `Deliverable1Dataset_CLEANED.csv`

**To run:**
```bash
python3 clean_dataset_pipeline.py
```

---

## References

**Missing Data Classification:**
- Rubin, D. B. (1976). Inference and missing data. Biometrika, 63(3), 581-592.
- Little, R. J., & Rubin, D. B. (2019). Statistical analysis with missing data (Vol. 793). John Wiley & Sons.

**Imputation Methods:**
- Schafer, J. L., & Graham, J. W. (2002). Missing data: our view of the state of the art. Psychological methods, 7(2), 147.

---

**Analysis Date:** October 2, 2025  
**Dataset:** Deliverable1Dataset.csv  
**Final Output:** Deliverable1Dataset_CLEANED.csv

