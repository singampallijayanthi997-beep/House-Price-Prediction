from flask import Flask, render_template, request, redirect
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

# ============================================================
# CREATE FLASK APP
# ============================================================

app = Flask(
    __name__,
    template_folder="../templates"
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

        formatted_integer = ",".join(groups) + "," + last_three

    return f"₹{formatted_integer}.{decimal_part}"

# ============================================================
# LOAD MODEL
# ============================================================

model = xgb.XGBRegressor()
model.load_model("model/xgboost_house_price_model.json")

encoder = joblib.load("model/one_hot_encoder.pkl")
preprocessing_info = joblib.load("model/preprocessing_info.pkl")

numerical_columns = preprocessing_info["numerical_columns"]
categorical_columns = preprocessing_info["categorical_columns"]

print("Model loaded successfully!")

# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():
    return redirect("/predict")

# ============================================================
# PREDICTION PAGE (GET + POST)
# ============================================================

@app.route("/predict", methods=["GET", "POST"])
def predict():

    # -------------------------
    # OPEN FORM
    # -------------------------
    if request.method == "GET":
        return render_template("index.html")

    # -------------------------
    # GET FORM DATA
    # -------------------------
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

    # -------------------------
    # VALIDATION
    # -------------------------
    if not 1 <= house["OverallQual"] <= 10:
        return render_template("index.html", error="Overall Quality must be between 1 and 10.")

    if house["GrLivArea"] <= 0:
        return render_template("index.html", error="Living Area must be greater than 0.")

    # -------------------------
    # DATAFRAME
    # -------------------------
    house_df = pd.DataFrame([house])

    for col in numerical_columns:
        if col not in house_df.columns:
            house_df[col] = 0

    for col in categorical_columns:
        if col not in house_df.columns:
            house_df[col] = "None"

    house_df = house_df[list(numerical_columns) + list(categorical_columns)]

    # -------------------------
    # ENCODING
    # -------------------------
    encoded = encoder.transform(house_df[categorical_columns])

    numerical = house_df[numerical_columns].to_numpy()

    final_data = np.hstack([numerical, encoded])

    # -------------------------
    # PREDICTION
    # -------------------------
    predicted_price = model.predict(final_data)[0]

    formatted_price = format_indian_currency(predicted_price)

    # -------------------------
    # RETURN RESULT
    # -------------------------
    return render_template(
        "index.html",
        prediction=formatted_price
    )

# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)