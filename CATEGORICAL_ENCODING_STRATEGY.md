# Categorical Encoding Strategy & Order

## Dataset Overview
- **Source:** `output_data/4_discount_applied/final_cleaned_dataset.csv`
- **Total Rows:** 11,971 (after missing data handling)
- **Categorical Attributes:** 7 (excluding Transaction ID)

---

## Categorical Attributes Summary

| Attribute | Unique Values | Cardinality Level | Data Type |
|-----------|--------------|-------------------|-----------|
| Transaction ID | 11,971 | Very High | Identifier (exclude) |
| Customer ID | 25 | Moderate-High | Identifier |
| Category | 10 | Moderate | Nominal |
| Item | 250 | Very High | Nominal |
| Payment Method | 3 | Low | Nominal |
| Location | 2 | Very Low | Binary |
| Discount Applied | 3 | Low | Nominal |

---

## Encoding Order & Priority

### **⚠️ CRITICAL: Data Source for Each Step**

**For EVERY encoding step (1-6), always start with the SAME source dataset:**
- **Source File:** `Handle missing data/output_data/4_discount_applied/final_cleaned_dataset.csv`

**DO NOT use the output of one encoding step as input for the next step.**

**Why:**
- Each encoding step is an **independent transformation** of the original categorical columns
- Using sequential outputs would cause cumulative modifications and data inconsistencies
- All steps operate on the same cleaned dataset in parallel workflows
- Only Step 7 (final combination) merges all encoded features together

**Workflow Pattern:**
```
Step 1: final_cleaned_dataset.csv → Customer ID encoding → output_1.csv
Step 2: final_cleaned_dataset.csv → Location encoding → output_2.csv
Step 3: final_cleaned_dataset.csv → Payment encoding → output_3.csv
...
Step 7: Combine encoded columns from all outputs → final_encoded.csv
```

---

### **PRIORITY ORDER (Handle in this sequence):**

#### **STEP 1: Customer ID - Target Encoding** ✓ **HANDLE FIRST**

**Encoding Method:** Target Encoding (using `Total Spent` as target)

**Why Handle First:**
1. **Foundation for other encodings:** Target encoding creates a numerical representation that can be used in subsequent analysis
2. **Prevents data leakage:** Calculating target encoding first ensures we use the full dataset before any subsetting
3. **High information value:** Captures customer spending behavior patterns that are critical for transaction analysis
4. **Required by assignment:** Question 1c specifically asks for this encoding with Total Spent as target

**Why Target Encoding (and not other methods):**

| Alternative Method | Why NOT Suitable |
|-------------------|------------------|
| **One-Hot Encoding** | Would create 25 columns → sparse, high-dimensional, inefficient |
| **Label Encoding** | Implies ordinal relationship (CUST_01 < CUST_02) which doesn't exist |
| **Frequency Encoding** | Only captures transaction count, not spending power/behavior |
| **Ordinal Encoding** | Same issue as Label - no natural order among customers |

**Technical Implementation:**
```python
# Calculate mean Total Spent per Customer ID
customer_target_encoding = df.groupby('Customer ID')['Total Spent'].mean()
df['Customer_ID_Encoded'] = df['Customer ID'].map(customer_target_encoding)
```

**Why This Works:**
- Customers who spend more get higher encoded values
- Captures behavioral patterns (high-value vs. low-value customers)
- Single column preserves dimensionality
- Numerical output compatible with all ML algorithms

**Expected Output:**
- New column: `Customer_ID_Encoded` (continuous numerical)
- Range: Min/Max of average spending per customer
- Example: CUST_01 → 127.45 (avg spend), CUST_25 → 89.30 (avg spend)

**File Output:** `output_data/5_encoding/customer_id_target_encoded.csv`

---

#### **STEP 2: Location - Binary Encoding** ✓ **HANDLE SECOND**

**Encoding Method:** Binary/Label Encoding (0/1)

**Why Handle Second:**
1. **Simplest encoding:** Only 2 unique values → quickest to process
2. **No dependencies:** Doesn't depend on other columns
3. **Low risk:** Binary encoding is unambiguous and error-free
4. **Foundation feature:** Location might correlate with other attributes (discount patterns, payment methods)

**Why Binary Encoding (and not other methods):**

| Alternative Method | Why NOT Suitable |
|-------------------|------------------|
| **One-Hot Encoding** | Overkill for binary variable; creates 2 columns when 1 is sufficient |
| **Target Encoding** | Unnecessarily complex for binary categorical |
| **Frequency Encoding** | Loses the actual categorical meaning (In-store vs Online) |

**Technical Implementation:**
```python
# In-store = 0, Online = 1
df['Location_Encoded'] = (df['Location'] == 'Online').astype(int)
```

**Why This Works:**
- Binary variables naturally map to 0/1
- Preserves the "online vs. offline" distinction
- Single column (efficient)
- Interpretable: 1 = Online, 0 = In-store

**Expected Output:**
- New column: `Location_Encoded` (binary: 0 or 1)
- Distribution preserved from original Location column

**File Output:** `output_data/5_encoding/location_binary_encoded.csv`

---

#### **STEP 3: Payment Method - One-Hot Encoding** ✓ **HANDLE THIRD**

**Encoding Method:** One-Hot Encoding

**Why Handle Third:**
1. **Low cardinality:** Only 3 categories → manageable
2. **No ordinal relationship:** Cash, Credit Card, Digital Wallet are independent choices
3. **Standard approach:** Industry best practice for low-cardinality nominal variables
4. **Prepares for analysis:** Many algorithms require numerical input

**Why One-Hot Encoding (and not other methods):**

| Alternative Method | Why NOT Suitable |
|-------------------|------------------|
| **Label Encoding** | Implies Cash(0) < Credit(1) < Digital(2) → false ordering |
| **Target Encoding** | Overcomplicates simple 3-category variable; risk of overfitting |
| **Frequency Encoding** | Loses categorical identity; only shows popularity |
| **Binary Encoding** | More than 2 categories → can't use simple 0/1 |

**Technical Implementation:**
```python
payment_encoded = pd.get_dummies(df['Payment Method'], prefix='Payment', drop_first=False)
# Creates: Payment_Cash, Payment_Credit Card, Payment_Digital Wallet
```

**Why This Works:**
- Each payment method becomes independent binary column
- No false ordinal relationships introduced
- Standard for nominal categorical with low cardinality
- All algorithms can process binary columns

**Expected Output:**
- 3 new columns: `Payment_Cash`, `Payment_Credit Card`, `Payment_Digital Wallet`
- Each row has exactly one "1" and two "0"s

**File Output:** `output_data/5_encoding/payment_method_one_hot_encoded.csv`

---

#### **STEP 4: Discount Applied - One-Hot Encoding** ✓ **HANDLE FOURTH**

**Encoding Method:** One-Hot Encoding

**Why Handle Fourth:**
1. **3 distinct categories:** True, False, Unknown (handled as separate category from missing data phase)
2. **Nominal variable:** No order between discount states
3. **Unknown is meaningful:** Represents MCAR pattern, should be preserved as category
4. **Standard treatment:** One-hot encoding treats all 3 equally

**Why One-Hot Encoding (and not other methods):**

| Alternative Method | Why NOT Suitable |
|-------------------|------------------|
| **Label Encoding** | False ordering: False(0) < True(1) < Unknown(2) makes no sense |
| **Binary (0/1)** | Can't represent 3 categories with single binary column |
| **Target Encoding** | Overly complex; discount is already related to Total Spent |
| **Ordinal Encoding** | No natural order exists between True/False/Unknown |

**Technical Implementation:**
```python
discount_encoded = pd.get_dummies(df['Discount Applied'], prefix='Discount', drop_first=False)
# Creates: Discount_True, Discount_False, Discount_Unknown
```

**Why This Works:**
- Treats "Unknown" as legitimate category (not missing data)
- No false relationships introduced
- Preserves MCAR information from missing data analysis
- Each state is independent

**Expected Output:**
- 3 new columns: `Discount_True`, `Discount_False`, `Discount_Unknown`
- Each row has exactly one "1" and two "0"s
- Distribution: ~33% each (from missing data analysis)

**File Output:** `output_data/5_encoding/discount_applied_one_hot_encoded.csv`

---

#### **STEP 5: Category - One-Hot Encoding** ✓ **HANDLE FIFTH**

**Encoding Method:** One-Hot Encoding

**Why Handle Fifth:**
1. **Moderate cardinality:** 10 categories is manageable for one-hot
2. **Core feature:** Product category is critical for analysis but doesn't need early encoding
3. **No dependencies:** Can be encoded independently after simpler features
4. **Standard approach:** Industry norm for product categories

**Why One-Hot Encoding (and not other methods):**

| Alternative Method | Why NOT Suitable |
|-------------------|------------------|
| **Label Encoding** | Implies false order: Beverages(0) < Food(1) < Furniture(2) |
| **Target Encoding** | Risk of overfitting; categories might correlate with Total Spent |
| **Frequency Encoding** | Loses category identity; only captures popularity |
| **Binary Encoding** | Requires 4 binary columns (less interpretable than 10 one-hot) |

**Technical Implementation:**
```python
category_encoded = pd.get_dummies(df['Category'], prefix='Category', drop_first=False)
# Creates 10 columns: Category_Beverages, Category_Food, Category_Furniture, etc.
```

**Why This Works:**
- Each category becomes independent binary feature
- No false relationships between unrelated categories
- Interpretable: Category_Food=1 means "this is a food transaction"
- 10 columns is acceptable dimensionality

**Expected Output:**
- 10 new columns (one per category)
- Categories include: Beverages, Butchers, Computers and electric accessories, Electric household essentials, Food, Furniture, Milk Products, Patisserie

**File Output:** `output_data/5_encoding/category_one_hot_encoded.csv`

---

#### **STEP 6: Item - Frequency Encoding** ✓ **HANDLE SIXTH**

**Encoding Method:** Frequency Encoding

**Why Handle Sixth (Last):**
1. **Very high cardinality:** 250 unique items → most complex encoding
2. **Least critical for initial analysis:** Item details less important than category-level patterns
3. **Requires careful consideration:** Multiple encoding strategies possible
4. **Dimensionality concerns:** One-hot would create 250 columns (impractical)

**Why Frequency Encoding (and not other methods):**

| Alternative Method | Why NOT Suitable |
|-------------------|------------------|
| **One-Hot Encoding** | Creates 250 columns → extremely sparse, memory-intensive, curse of dimensionality |
| **Label Encoding** | False ordering: Item_1(0) < Item_2(1) < ... < Item_250(249) meaningless |
| **Target Encoding** | Risk of severe overfitting with 250 categories; data leakage concerns |
| **Ordinal Encoding** | Same issue as Label - no natural order among items |
| **Binary Encoding** | Requires 8 binary columns (2^8=256); less interpretable |

**Technical Implementation:**
```python
# Calculate frequency (proportion) of each item
item_frequency = df['Item'].value_counts(normalize=True)
df['Item_Encoded'] = df['Item'].map(item_frequency)
```

**Why This Works:**
- Captures item popularity (frequently bought items get higher values)
- Single column preserves dimensionality
- Items with same frequency treated similarly
- Reduces 250 categories to continuous variable [0, 1]

**Why NOT One-Hot (Primary Concern):**
- **Sparsity:** 250 columns where only 1 is "1" per row → 99.6% zeros
- **Memory:** 250x storage vs. 1 column
- **Curse of dimensionality:** Most ML algorithms perform poorly with 250 features for ~12K rows
- **Interpretability:** 250 coefficients difficult to interpret

**Alternative Considered - Target Encoding:**
- **Pros:** Captures item spending patterns
- **Cons:**
  - Risk of overfitting (250 categories)
  - Data leakage if not using proper cross-validation
  - Item prices already captured in "Price Per Unit" column

**Expected Output:**
- New column: `Item_Encoded` (continuous, range [0, max_frequency])
- Popular items (e.g., Item_16_MILK) → higher values
- Rare items → lower values

**File Output:** `output_data/5_encoding/item_frequency_encoded.csv`

---

## Final Step: Combine All Encodings

**STEP 7: Create Fully Encoded Dataset**

**Method:** Merge all encoded features, drop original categorical columns

**Columns to KEEP:**
- Transaction ID (identifier only, not for analysis)
- All numerical features: Price Per Unit, Quantity, Total Spent, Transaction Date
- All encoded features (19 new columns)

**Columns to DROP:**
- Original categorical columns: Customer ID, Category, Item, Payment Method, Location, Discount Applied

**Technical Implementation:**
```python
# Combine all encoded columns
final_df = df[[
    'Transaction ID',  # Keep for tracking
    'Price Per Unit', 'Quantity', 'Total Spent',  # Numerical features
    'Customer_ID_Encoded',  # From Step 1
    'Location_Encoded',  # From Step 2
    'Payment_Cash', 'Payment_Credit Card', 'Payment_Digital Wallet',  # From Step 3
    'Discount_True', 'Discount_False', 'Discount_Unknown',  # From Step 4
    'Category_Beverages', 'Category_Butchers', ...,  # From Step 5 (10 columns)
    'Item_Encoded'  # From Step 6
]]
```

**File Output:** `output_data/5_encoding/final_fully_encoded_dataset.csv`

---

## Summary: Why This Order?

### **Logical Progression:**

1. **Customer ID (Step 1)** → Required by assignment; foundation for analysis
2. **Location (Step 2)** → Simplest; no dependencies
3. **Payment Method (Step 3)** → Low complexity; standard one-hot
4. **Discount Applied (Step 4)** → Low complexity; preserves MCAR handling
5. **Category (Step 5)** → Moderate complexity; after simpler features
6. **Item (Step 6)** → Most complex; requires dimensionality reduction

### **Key Principles:**

✅ **Start with requirements** (Customer ID target encoding - question 1c)
✅ **Simple before complex** (Binary → Low-cardinality → High-cardinality)
✅ **Avoid dimensionality explosion** (Frequency encoding for 250 items, not one-hot)
✅ **Preserve information** (One-hot for nominal, Binary for binary, Target for identifiers)
✅ **Prevent overfitting** (Avoid target encoding for high-cardinality non-identifiers)
✅ **Maintain interpretability** (Clear naming, logical structure)

---

## Final Encoded Dataset Structure

| Feature Category | Column Count | Encoding Method |
|-----------------|--------------|----------------|
| **Identifiers** | 1 | None (Transaction ID kept) |
| **Numerical** | 3 | None (already numerical) |
| **Customer** | 1 | Target Encoding |
| **Location** | 1 | Binary Encoding |
| **Payment** | 3 | One-Hot Encoding |
| **Discount** | 3 | One-Hot Encoding |
| **Category** | 10 | One-Hot Encoding |
| **Item** | 1 | Frequency Encoding |
| **TOTAL** | **23 columns** | Mixed Strategy |

**Compared to naive all-one-hot approach:** 23 columns vs. 291 columns (92% reduction)

---

## Validation Checklist

After encoding, verify:

```python
# 1. No missing values in encoded columns
assert df[encoded_columns].isna().sum().sum() == 0

# 2. One-hot columns sum to 1 per row
assert (df[['Payment_Cash', 'Payment_Credit Card', 'Payment_Digital Wallet']].sum(axis=1) == 1).all()
assert (df[['Discount_True', 'Discount_False', 'Discount_Unknown']].sum(axis=1) == 1).all()

# 3. Binary encoding has only 0/1
assert df['Location_Encoded'].isin([0, 1]).all()

# 4. Frequency encoding in [0, 1]
assert (df['Item_Encoded'] >= 0).all() and (df['Item_Encoded'] <= 1).all()

# 5. Row count unchanged
assert len(final_df) == 11971
```

---

**Report Generated:** Categorical encoding strategy for Deliverable 1 - Questions 1b, 1c, 1d