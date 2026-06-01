import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. Load data
print("🔄 Loading student performance dataset for comparison...")
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

# Split data (80% training, 20% validation)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Identify column types automatically
cat_cols = X.select_dtypes(include='object').columns
num_cols = X.select_dtypes(exclude='object').columns

# Preprocessing strategy
preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", SimpleImputer(strategy="mean"), num_cols)
])

# Define the 3 models to benchmark
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost Regressor": XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
}

results = []

print("\n" + "="*65)
print(" 🚀 RUNNING MODEL BENCHMARKING EXPERIMENTS ")
print("="*65)

# Train and calculate proper regression metrics for each model
for name, algorithm in models.items():
    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("model", algorithm)
    ])
    
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    
    # Calculate performance metrics
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    results.append({
        "Model": name,
        "R2 Score": f"{r2:.4f}",
        "MAE": f"{mae:.4f}",
        "MSE": f"{mse:.4f}",
        "RMSE": f"{rmse:.4f}"
    })

# Render comparison dashboard 
summary_df = pd.DataFrame(results)
print("\n", summary_df.to_string(index=False))
print("="*65 + "\n")