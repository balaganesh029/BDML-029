
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error


df = pd.read_csv("used_cars.csv")


required_columns = [
    "brand",
    "model",
    "year",
    "km_driven",
    "fuel",
    "transmission",
    "engine",
    "mileage",
    "max_power",
    "seats",
    "owner",
    "selling_price"
]

df = df[required_columns]


numeric_columns = [
    "year",
    "km_driven",
    "engine",
    "mileage",
    "max_power",
    "seats"
]


categorical_columns = [
    "brand",
    "model",
    "fuel",
    "transmission",
    "owner"
]


for column in numeric_columns:
    df[column] = (
        df[column]
        .astype(str)
        .str.extract(r"([-+]?\d*\.?\d+)")[0]
    )

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


df["selling_price"] = (
    df["selling_price"]
    .astype(str)
    .str.replace(",", "", regex=False)
)

df["selling_price"] = pd.to_numeric(
    df["selling_price"],
    errors="coerce"
)


for column in categorical_columns:
    df[column] = (
        df[column]
        .astype(str)
        .str.strip()
    )

    df[column] = df[column].replace(
        ["nan", "None", ""],
        np.nan
    )


df = df.dropna(
    subset=["selling_price"]
)


X = df.drop(
    "selling_price",
    axis=1
)

y = df["selling_price"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])


categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])


preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_columns
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_columns
    )
])


linear_model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        LinearRegression()
    )
])


random_forest_model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
    )
])


print("Training Linear Regression...")

linear_model.fit(
    X_train,
    y_train
)


print("Training Random Forest...")

random_forest_model.fit(
    X_train,
    y_train
)


linear_predictions = linear_model.predict(
    X_test
)

rf_predictions = random_forest_model.predict(
    X_test
)


linear_r2 = r2_score(
    y_test,
    linear_predictions
)

rf_r2 = r2_score(
    y_test,
    rf_predictions
)


linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)


print()
print("=" * 45)
print("MODEL PERFORMANCE")
print("=" * 45)

print()
print("Linear Regression")
print("R2 Score :", round(linear_r2, 4))
print("MAE      :", round(linear_mae, 2))

print()
print("Random Forest")
print("R2 Score :", round(rf_r2, 4))
print("MAE      :", round(rf_mae, 2))

print()
print("=" * 45)


os.makedirs(
    "models",
    exist_ok=True
)


joblib.dump(
    linear_model,
    "models/linear_regression.pkl"
)

joblib.dump(
    random_forest_model,
    "models/random_forest.pkl"
)


print()
print("Models saved successfully.")
print("Owner field handled as categorical data.")
