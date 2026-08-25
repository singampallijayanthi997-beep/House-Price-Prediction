from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os

# ============================================================
# CREATE FLASK APP
# ============================================================

app = Flask(
    __name__,
    template_folder="../templates"
)

# ============================================================
# PROJECT PATHS (WORKS LOCALLY + RENDER)
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "xgboost_house_price_model.json"
)

ENCODER_PATH = os.path.join(
    BASE_DIR,
    "model",
    "one_hot_encoder.pkl"
)

PREPROCESS_PATH = os.path.join(
    BASE_DIR,
    "model",
    "preprocessing_info.pkl"
)

# ============================================================
# INDIAN CURRENCY FORMATTER
# ============================================================

def format_indian_currency(value):

    value = float(value)

    integer_part, decimal_part = f"{value:.2f}".split(".")

    if len(integer_part) <= 3:

        formatted_integer = integer_part

    else:

        last_three = integer_part[-3:]
        remaining = integer_part[:-3]

        groups = []

        while remaining:
            groups.insert(0, remaining[-2:])
            remaining = remaining[:-2]

        formatted_integer = (
            ",".join(groups)
            + ","
            + last_three
        )

    return f"₹{formatted_integer}.{decimal_part}"

# ============================================================
# LOAD XGBOOST MODEL
# ============================================================

model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)

print("XGBoost model loaded successfully!")

# ============================================================
# LOAD ONE-HOT ENCODER
# ============================================================

encoder = joblib.load(ENCODER_PATH)

print("One-Hot Encoder loaded successfully!")

# ============================================================
# LOAD PREPROCESSING INFORMATION
# ============================================================

preprocessing_info = joblib.load(PREPROCESS_PATH)

numerical_columns = preprocessing_info["numerical_columns"]
categorical_columns = preprocessing_info["categorical_columns"]

print("Preprocessing information loaded successfully!")

print("Number of numerical columns:", len(numerical_columns))
print("Number of categorical columns:", len(categorical_columns))

# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template("index.html")

# ============================================================
# PREDICTION ROUTE
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    # ========================================================
    # GET VALUES FROM HTML FORM
    # ========================================================

    house = {

        "OverallQual": int(request.form["OverallQual"]),
        "GrLivArea": int(request.form["GrLivArea"]),
        "GarageCars": int(request.form["GarageCars"]),
        "GarageArea": int(request.form["GarageArea"]),
        "TotalBsmtSF": int(request.form["TotalBsmtSF"]),
        "1stFlrSF": int(request.form["1stFlrSF"]),
        "FullBath": int(request.form["FullBath"]),
        "TotRmsAbvGrd": int(request.form["TotRmsAbvGrd"]),
        "YearBuilt": int(request.form["YearBuilt"]),
        "Neighborhood": request.form["Neighborhood"],
        "KitchenQual": request.form["KitchenQual"]

    }

    print("\nHouse details received:")
    print(house)

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    if not 1 <= house["OverallQual"] <= 10:
        return render_template(
            "index.html",
            error="Overall Quality must be between 1 and 10."
        )

    if house["GrLivArea"] <= 0:
        return render_template(
            "index.html",
            error="Living Area must be greater than 0."
        )

    if not 0 <= house["GarageCars"] <= 5:
        return render_template(
            "index.html",
            error="Garage Cars must be between 0 and 5."
        )

    if house["GarageArea"] < 0:
        return render_template(
            "index.html",
            error="Garage Area cannot be negative."
        )

    if house["TotalBsmtSF"] < 0:
        return render_template(
            "index.html",
            error="Basement Area cannot be negative."
        )

    if house["1stFlrSF"] <= 0:
        return render_template(
            "index.html",
            error="First Floor Area must be greater than 0."
        )

    if not 0 <= house["FullBath"] <= 5:
        return render_template(
            "index.html",
            error="Full Bathrooms must be between 0 and 5."
        )

    if not 1 <= house["TotRmsAbvGrd"] <= 20:
        return render_template(
            "index.html",
            error="Total Rooms must be between 1 and 20."
        )

    if not 1800 <= house["YearBuilt"] <= 2026:
        return render_template(
            "index.html",
            error="Year Built must be between 1800 and 2026."
        )

    # ========================================================
    # CONVERT TO DATAFRAME
    # ========================================================

    house_df = pd.DataFrame([house])

    print("House converted to DataFrame successfully!")

    # ========================================================
    # ADD MISSING NUMERICAL COLUMNS
    # ========================================================

    for column in numerical_columns:
        if column not in house_df.columns:
            house_df[column] = 0

    # ========================================================
    # ADD MISSING CATEGORICAL COLUMNS
    # ========================================================

    for column in categorical_columns:
        if column not in house_df.columns:
            house_df[column] = "None"

    # ========================================================
    # ARRANGE COLUMNS
    # ========================================================

    house_df = house_df[
        list(numerical_columns)
        + list(categorical_columns)
    ]

    print("House columns prepared successfully!")
    print("House shape:", house_df.shape)

    # ========================================================
    # CHECK MISSING VALUES
    # ========================================================

    missing_values = house_df.isnull().sum().sum()

    print("Missing values in new house:", missing_values)

    # ========================================================
    # ENCODE CATEGORICAL DATA
    # ========================================================

    house_encoded = encoder.transform(
        house_df[categorical_columns]
    )

    print("New house categorical data encoded successfully!")
    print("Encoded house shape:", house_encoded.shape)

    # ========================================================
    # GET NUMERICAL DATA
    # ========================================================

    house_numerical = house_df[
        numerical_columns
    ].to_numpy()

    # ========================================================
    # COMBINE NUMERICAL + ENCODED
    # ========================================================

    house_final = np.hstack([
        house_numerical,
        house_encoded
    ])

    print("New house data prepared for prediction!")
    print("Final house shape:", house_final.shape)

    # ========================================================
    # MAKE PREDICTION
    # ========================================================

    predicted_price = model.predict(house_final)[0]

    print("Predicted House Price:", predicted_price)

    # ========================================================
    # FORMAT PRICE
    # ========================================================

    formatted_price = format_indian_currency(predicted_price)

    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    return render_template(
        "index.html",
        prediction=formatted_price
    )

# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":

    app.run(debug=True)