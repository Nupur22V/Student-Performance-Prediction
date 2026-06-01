import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import joblib

print("🔄 Initializing production model training...")

# 1. Load data
df = pd.read_csv("cleaned_student_performance_dataset.csv")

X = df[[
    'study_hours(per_day)',
    'sleep_hours',
    'attendance(%)',
    'coding_hours(per_day)',
    'social_media_usage',
    'participation_in_class'
]]
y = df['cgpa']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessing setup
cat_cols = X.select_dtypes(include='object').columns
num_cols = X.select_dtypes(exclude='object').columns

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", SimpleImputer(strategy="mean"), num_cols)
])

# 2. Build production pipeline with our best-performing model (Linear Regression)
production_pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", LinearRegression())
])

# Train on our training set
print("🧠 Training production model...")
production_pipeline.fit(X_train, y_train)

# 3. Save model.pkl directly to the current main project folder
model_filename = "model.pkl"
joblib.dump(production_pipeline, model_filename)

print("="*60)
print(f"💾 Successfully saved production model directly to: '{model_filename}'")
print("🎉 Your backend web server (flash.py) is ready to deploy!")
print("="*60)