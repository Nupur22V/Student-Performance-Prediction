

import pandas as pd
import numpy as np

# =========================
# 1. Load Dataset
# =========================
file_path = "Raw data.csv"   # change path if needed
df = pd.read_csv(file_path)

# =========================
# 2. Basic Exploration
# =========================
print("Shape:", df.shape)
print("\nColumns:\n", df.columns)
print("\nInfo:\n")
df.info()

print("\nMissing Values:\n", df.isnull().sum())

# =========================
# 3. Standardize Column Names
# =========================
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# =========================
# 4. Remove Duplicates
# =========================
df = df.drop_duplicates()

# =========================
# 5. Handle Missing Values
# =========================

for col in df.columns:
    if df[col].dtype == "object":
        # categorical → fill with mode
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        # numeric → fill with mean
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mean(), inplace=True)

# =========================
# 6. Remove Outliers (IQR method)
# =========================
numeric_cols = df.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# =========================
# 7. Reset Index
# =========================
df.reset_index(drop=True, inplace=True)

# =========================
# 8. Final Check
# =========================
print("\nFinal Shape:", df.shape)
print("\nRemaining Missing Values:\n", df.isnull().sum())

# =========================
# 9. Save Clean Dataset
# =========================
output_path = "cleaned_student_performance_dataset.csv"
df.to_csv(output_path, index=False)

print("\n✅ Clean dataset saved successfully!")