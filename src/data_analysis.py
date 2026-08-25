import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error
)

from xgboost import XGBRegressor


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("dataset/train.csv")

print("\n========== DATASET ==========")
print(df.head())


# ============================================================
# 2. UNDERSTAND THE DATA
# ============================================================

print("\n========== DATA INFORMATION ==========")
print(df.info())


# ============================================================
# 3. CHECK MISSING VALUES
# ============================================================

print("\n========== MISSING VALUES ==========")

missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])


# ============================================================
# 4. SALE PRICE STATISTICS
# ============================================================

print("\n========== SALE PRICE STATISTICS ==========")
print(df["SalePrice"].describe())


# ============================================================
# 5. HOUSE PRICE DISTRIBUTION
# ============================================================

plt.hist(df["SalePrice"], bins=30)

plt.xlabel("House Price")
plt.ylabel("Number of Houses")
plt.title("Distribution of House Prices")

plt.show()


# ============================================================
# 6. LIVING AREA VS HOUSE PRICE
# ============================================================

plt.scatter(df["GrLivArea"], df["SalePrice"])

plt.xlabel("Living Area")
plt.ylabel("Sale Price")
plt.title("Living Area vs House Price")

plt.show()


# ============================================================
# 7. CORRELATION WITH SALE PRICE
# ============================================================

correlation = (
    df.corr(numeric_only=True)["SalePrice"]
    .sort_values(ascending=False)
)

print("\n========== TOP CORRELATIONS ==========")
print(correlation.head(10))


# ============================================================
# 8. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

print("\n========== FEATURES AND TARGET ==========")
print("Features shape:", X.shape)
print("Target shape:", y.shape)


# ============================================================
# 9. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n========== TRAIN-TEST SPLIT ==========")
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)


# ============================================================
# 10. IDENTIFY NUMERICAL AND CATEGORICAL COLUMNS
# ============================================================

numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_columns = X.select_dtypes(
    include=["str", "object"]
).columns

print("\n========== COLUMN TYPES ==========")
print("Number of numerical columns:", len(numerical_columns))
print("Number of categorical columns:", len(categorical_columns))


# ============================================================
# 11. HANDLE MISSING NUMERICAL VALUES
# ============================================================

for column in numerical_columns:

    # Use training median
    median_value = X_train[column].median()

    X_train[column] = X_train[column].fillna(median_value)
    X_test[column] = X_test[column].fillna(median_value)


# ============================================================
# 12. HANDLE MISSING CATEGORICAL VALUES
# ============================================================

for column in categorical_columns:

    # Use most frequent value from training data
    mode_value = X_train[column].mode()[0]

    X_train[column] = X_train[column].fillna(mode_value)
    X_test[column] = X_test[column].fillna(mode_value)


print("\n========== AFTER CLEANING ==========")
print(
    "Missing values in training data:",
    X_train.isnull().sum().sum()
)

print(
    "Missing values in testing data:",
    X_test.isnull().sum().sum()
)


# ============================================================
# 13. ONE-HOT ENCODING
# ============================================================

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

print("\nOne-Hot Encoder created successfully!")


# ============================================================
# 14. ENCODE CATEGORICAL DATA
# ============================================================

X_train_encoded = encoder.fit_transform(
    X_train[categorical_columns]
)

X_test_encoded = encoder.transform(
    X_test[categorical_columns]
)

print("\n========== ENCODING ==========")
print("Categorical data encoded successfully!")
print("Encoded training shape:", X_train_encoded.shape)
print("Encoded testing shape:", X_test_encoded.shape)


# ============================================================
# 15. GET NUMERICAL DATA
# ============================================================

X_train_numerical = X_train[
    numerical_columns
].to_numpy()

X_test_numerical = X_test[
    numerical_columns
].to_numpy()


# ============================================================
# 16. COMBINE NUMERICAL + CATEGORICAL DATA
# ============================================================

X_train_final = np.hstack([
    X_train_numerical,
    X_train_encoded
])

X_test_final = np.hstack([
    X_test_numerical,
    X_test_encoded
])

print("\n========== FINAL DATA ==========")
print("Final training shape:", X_train_final.shape)
print("Final testing shape:", X_test_final.shape)


# ============================================================
# 17. RANDOM FOREST MODEL
# ============================================================

print("\n========== RANDOM FOREST ==========")

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train_final,
    y_train
)

print("Model trained successfully!")


# ============================================================
# 18. RANDOM FOREST PREDICTIONS
# ============================================================

y_pred = model.predict(X_test_final)

print("Predictions made successfully!")

print("First 5 predictions:")
print(y_pred[:5])


# ============================================================
# 19. RANDOM FOREST EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)

rmse = root_mean_squared_error(
    y_test,
    y_pred
)

print("\n========== RANDOM FOREST RESULTS ==========")
print("Mean Absolute Error:", mae)
print("R² Score:", r2)
print("RMSE:", rmse)


# ============================================================
# 20. XGBOOST MODEL
# ============================================================

print("\n========== XGBOOST ==========")

xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)


# ============================================================
# 21. TRAIN XGBOOST
# ============================================================

xgb_model.fit(
    X_train_final,
    y_train
)

print("XGBoost model trained successfully!")


# ============================================================
# 22. XGBOOST PREDICTIONS
# ============================================================

xgb_pred = xgb_model.predict(
    X_test_final
)

print("XGBoost predictions made successfully!")

print("First 5 XGBoost predictions:")
print(xgb_pred[:5])


# ============================================================
# 23. XGBOOST EVALUATION
# ============================================================

xgb_mae = mean_absolute_error(
    y_test,
    xgb_pred
)

xgb_r2 = r2_score(
    y_test,
    xgb_pred
)

xgb_rmse = root_mean_squared_error(
    y_test,
    xgb_pred
)

print("\n========== XGBOOST RESULTS ==========")
print("XGBoost MAE:", xgb_mae)
print("XGBoost R²:", xgb_r2)
print("XGBoost RMSE:", xgb_rmse)


# ============================================================
# 24. CHECK XGBOOST OVERFITTING
# ============================================================

xgb_train_pred = xgb_model.predict(
    X_train_final
)

xgb_train_r2 = r2_score(
    y_train,
    xgb_train_pred
)

print("\n========== OVERFITTING CHECK ==========")
print("XGBoost Training R²:", xgb_train_r2)
print("XGBoost Testing R²:", xgb_r2)


# ============================================================
# 25. MODEL COMPARISON
# ============================================================

print("\n========== MODEL COMPARISON ==========")

print("\nRandom Forest:")
print("MAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)

print("\nXGBoost:")
print("MAE :", xgb_mae)
print("RMSE:", xgb_rmse)
print("R²  :", xgb_r2)
print("\nXGBoost:")
print("MAE :", xgb_mae)
print("RMSE:", xgb_rmse)
print("R²  :", xgb_r2)


# ============================================================
# 26. SAVE XGBOOST MODEL
# ============================================================

xgb_model.save_model(
    "model/xgboost_house_price_model.json"
)

print("\nXGBoost model saved successfully!")


# ============================================================
# 27. SAVE ENCODER
# ============================================================

import joblib

joblib.dump(
    encoder,
    "model/one_hot_encoder.pkl"
)

print("Encoder saved successfully!")
# ============================================================
# 28. SAVE PREPROCESSING INFORMATION
# ============================================================

preprocessing_info = {
    "numerical_columns": list(numerical_columns),
    "categorical_columns": list(categorical_columns)
}

joblib.dump(
    preprocessing_info,
    "model/preprocessing_info.pkl"
)

print("Preprocessing information saved successfully!")