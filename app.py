from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

linear_model = joblib.load("models/linear_regression.pkl")
random_forest_model = joblib.load("models/random_forest.pkl")

df = pd.read_csv("used_cars.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Clean categorical columns
for column in ["brand", "model", "fuel", "transmission", "owner"]:
    df[column] = df[column].astype(str).str.strip()

brands = sorted(df["brand"].dropna().unique().tolist())
fuels = sorted(df["fuel"].dropna().unique().tolist())
transmissions = sorted(df["transmission"].dropna().unique().tolist())
owners = sorted(df["owner"].dropna().unique().tolist())


@app.route("/")
def home():
    return render_template(
        "index.html",
        brands=brands,
        fuels=fuels,
        transmissions=transmissions,
        owners=owners
    )


@app.route("/models/<path:brand>")
def get_models(brand):

    brand = brand.strip()

    print("Brand requested:", brand)

    matching_rows = df[
        df["brand"].str.strip().str.lower() == brand.lower()
    ]

    models = sorted(
        matching_rows["model"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    print("Models found:", models)

    return jsonify(models)


@app.route("/predict", methods=["POST"])
def predict():

    data = {
        "brand": request.form["brand"],
        "model": request.form["model"],
        "year": float(request.form["year"]),
        "km_driven": float(request.form["km_driven"]),
        "fuel": request.form["fuel"],
        "transmission": request.form["transmission"],
        "engine": float(request.form["engine"]),
        "mileage": float(request.form["mileage"]),
        "max_power": float(request.form["max_power"]),
        "seats": float(request.form["seats"]),
        "owner": request.form["owner"]
    }

    input_data = pd.DataFrame([data])

    linear_prediction = linear_model.predict(input_data)[0]

    rf_prediction = random_forest_model.predict(input_data)[0]

    final_prediction = (
        linear_prediction + rf_prediction
    ) / 2

    return render_template(
        "index.html",
        brands=brands,
        fuels=fuels,
        transmissions=transmissions,
        owners=owners,
        selected_brand=data["brand"],
        selected_model=data["model"],
        linear_price=round(linear_prediction),
        rf_price=round(rf_prediction),
        final_price=round(final_prediction),
        predicted=True
    )


if __name__ == "__main__":
    app.run(debug=True)