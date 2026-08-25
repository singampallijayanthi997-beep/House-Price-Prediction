import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
# Load the trained XGBoost model
model = xgb.XGBRegressor()

model.load_model(
    "model/xgboost_house_price_model.json"
)

print("XGBoost model loaded successfully!")


# Load the One-Hot Encoder
encoder = joblib.load(
    "model/one_hot_encoder.pkl"
)

print("One-Hot Encoder loaded successfully!")


# Load preprocessing information
preprocessing_info = joblib.load(
    "model/preprocessing_info.pkl"
)

numerical_columns = preprocessing_info["numerical_columns"]
categorical_columns = preprocessing_info["categorical_columns"]

print("Preprocessing information loaded successfully!")
print("Number of numerical columns:", len(numerical_columns))
print("Number of categorical columns:", len(categorical_columns))
# Create a sample house
house = {
    "OverallQual": 7,
    "GrLivArea": 1800,
    "GarageCars": 2,
    "GarageArea": 500,
    "TotalBsmtSF": 1000,
    "1stFlrSF": 1000,
    "FullBath": 2,
    "TotRmsAbvGrd": 7,
    "YearBuilt": 2005,
    "Neighborhood": "CollgCr",
    "KitchenQual": "Gd"
}

print("Sample house details created successfully!")
print(house)
# Convert house details into a DataFrame
house_df = pd.DataFrame([house])

print("House converted to DataFrame successfully!")
print(house_df)
# Add missing columns
for column in numerical_columns:
    if column not in house_df.columns:
        house_df[column] = 0

for column in categorical_columns:
    if column not in house_df.columns:
        house_df[column] = "None"

# Arrange columns in the original order
house_df = house_df[numerical_columns + categorical_columns]
print("House columns prepared successfully!")
print("House shape:", house_df.shape)
# Check for missing values
print("Missing values in new house:", house_df.isnull().sum().sum())
# Encode categorical columns of the new house
house_encoded = encoder.transform(
    house_df[categorical_columns]
)

print("New house categorical data encoded successfully!")
print("Encoded house shape:", house_encoded.shape)
# Get numerical data from the new house
house_numerical = house_df[numerical_columns].to_numpy()

# Combine numerical and encoded data
house_final = np.hstack([
    house_numerical,
    house_encoded
])

print("New house data prepared for prediction!")
print("Final house shape:", house_final.shape)
# Predict house price
predicted_price = model.predict(house_final)

print("Predicted House Price:", predicted_price[0])