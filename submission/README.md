## 📋 Project Overview

This submission contains a complete data analysis pipeline for handling missing data, encoding categorical variables, and rescaling numerical features for a retail transaction dataset.

---

## 📁 Project Structure

```
submission/
├── README.md                          # This file
├── datasource/
│   └── Deliverable1Dataset.csv       # Original dataset (12,575 rows)
│
├── sources/                           # Python source scripts (.py files)
│   ├── 1_handle_missing_data/        # 4 Python scripts
│   │   ├── total_spent_missing_data_2b.py
│   │   ├── price_per_unit_missing_data_2b.py
│   │   ├── item_missing_data_2b.py
│   │   └── discount_applied_missing_data_2b.py
│   │
│   ├── 2_handle_encoding_data/       # 6 Python scripts
│   │   ├── customer_id_encode_data_2d.py
│   │   ├── item_encode_data_2c.py
│   │   ├── category_encode_data_2c.py
│   │   ├── location_encode_data_2c.py
│   │   ├── payment_method_encode_data_2c.py
│   │   └── discount_applied_encode_data_2c.py
│   │
│   └── 3_handle_rescale_data/        # 4 Python scripts
│       ├── transaction_date_rescale_data_3c.py
│       ├── quantity_rescale_data_3c.py
│       ├── price_per_unit_rescale_data_3c.py
│       └── total_spent_rescale_data_3c.py
│
├── docs/                              # Jupyter notebooks (.ipynb files)
│   ├── 1_handle_missing_data/        # 4 notebooks with documentation
│   │   ├── total_spent_doc_missing_data_2_b.ipynb
│   │   ├── price_per_unit_doc_missing_data_2_b.ipynb
│   │   ├── item_missing_data_doc_2b.ipynb
│   │   └── discount_applied_doc_missing_data_2b.ipynb
│   │
│   ├── 2_handle_encoding_data/       # 7 notebooks with documentation
│   │   ├── customer_id_doc_encode_data_2_d.ipynb
│   │   ├── item_doc_encode_data_2_c.ipynb
│   │   ├── category_doc_encode_data_2_c.ipynb
│   │   ├── location_doc_encode_data_2_c.ipynb
│   │   ├── payment_method_doc_encode_data_2_c.ipynb
│   │   ├── discount_applied_doc_encode_data_2_c.ipynb
│   │   └── combine_all_doc.ipynb
│   │
│   └── 3_handle_rescale_data/        # 4 notebooks with documentation
│       ├── transaction_date_doc_rescale_data_3_c.ipynb
│       ├── quantity_rescale_data_3_c.ipynb
│       ├── price_per_unit_doc_rescale_data_3_c.ipynb
│       └── total_spent_doc_rescale_data_3_c.ipynb
│
└── output/                            # Generated output files
    ├── 1_handle_missing_data/        # Missing data outputs
    │   ├── total_spent_cleaned.csv
    │   ├── price_per_unit_reconstructed.csv
    │   ├── item_imputed.csv
    │   └── final_cleaned_dataset.csv  # ⭐ Input for Phase 2 & 3
    │
    ├── 2_handle_encoding_data/       # Encoding outputs
    │   ├── encoded_customer_id_dataset.csv
    │   ├── encoded_item_dataset.csv
    │   ├── encoded_category_dataset.csv
    │   ├── location_binary_encoded.csv
    │   ├── encoded_payment_method_dataset.csv
    │   ├── discount_applied_one_hot_encoded.csv
    │   └── final_fully_encoded_dataset.csv
    │
    └── 3_handle_rescale_data/        # Rescaling outputs
        ├── data_rescaling_norm_transaction_date.csv
        ├── data_rescaling_norm_quantity.csv
        ├── data_rescaling_std_quantity.csv
        ├── data_rescaling_robust_quantity.csv       # ⭐ Recommended
        ├── data_rescaling_norm_price_per_unit.csv   # ⭐ Recommended
        ├── data_rescaling_std_price_per_unit.csv
        ├── data_rescaling_robust_price_per_unit.csv
        ├── data_rescaling_norm_total_spent.csv
        ├── data_rescaling_std_total_spent.csv
        └── data_rescaling_robust_total_spent.csv    # ⭐ Recommended
```

---

