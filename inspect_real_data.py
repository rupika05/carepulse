import pandas as pd
import numpy as np

train_path = "data/train_ml03.csv"
test_path = "data/test_pred_ml03.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("=" * 60)
print("REAL COMPETITION DATASET INSPECTION")
print("=" * 60)

print(f"\nTrain Filename: {train_path}")
print(f"Train Shape: {train_df.shape[0]} rows x {train_df.shape[1]} columns")

print(f"\nTest Filename: {test_path}")
print(f"Test Shape: {test_df.shape[0]} rows x {test_df.shape[1]} columns")

print("\n--- Train Columns & Types ---")
for col in train_df.columns:
    print(f"  {col:<25} {str(train_df[col].dtype):<10} {train_df[col].nunique():>5} unique  {train_df[col].isnull().sum():>5} nulls")

print("\n--- Test Columns & Types ---")
for col in test_df.columns:
    print(f"  {col:<25} {str(test_df[col].dtype):<10} {test_df[col].nunique():>5} unique  {test_df[col].isnull().sum():>5} nulls")

print("\n--- First 3 rows of Train ---")
print(train_df.head(3).to_string())

print("\n--- First 3 rows of Test ---")
print(test_df.head(3).to_string())

# Find differences between train and test columns
diff_train = set(train_df.columns) - set(test_df.columns)
diff_test = set(test_df.columns) - set(train_df.columns)
print(f"\nColumns only in train: {diff_train}")
print(f"Columns only in test: {diff_test}")

# Identify Target
target_col = list(diff_train)[0] if len(diff_train) == 1 else "Unknown"
print(f"\nTarget Column identified: {target_col}")

if target_col in train_df.columns:
    print(f"\nTarget Class Distribution:")
    print(train_df[target_col].value_counts())

print(f"\nDuplicate rows in train: {train_df.duplicated().sum()}")
print(f"Duplicate rows in test: {test_df.duplicated().sum()}")
