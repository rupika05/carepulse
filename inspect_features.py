import pandas as pd

train = pd.read_csv("data/train_ml03.csv")
test = pd.read_csv("data/test_pred_ml03.csv")

print("--- NUMERICAL COLUMNS ---")
num_cols = train.select_dtypes(include=['number']).columns.tolist()
num_cols.remove('subject_id')
print(num_cols)

print("\n--- CATEGORICAL COLUMNS ---")
cat_cols = train.select_dtypes(include=['object']).columns.tolist()
cat_cols.remove('start_category')
print(cat_cols)

print("\n--- TARGET VALUE COUNTS ---")
print(train['start_category'].value_counts())
