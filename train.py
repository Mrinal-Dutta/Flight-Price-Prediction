
"""
Flight Price Prediction
Model Training Pipeline using XGBoost
"""

from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


# ==========================================================
# 1. PROJECT PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "Data_Train.xlsx"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "flight_price_prediction_pipeline.pkl"

RANDOM_STATE = 42


# ==========================================================
# 2. FEATURE DEFINITIONS
# ==========================================================

CATEGORICAL_FEATURES = [
    "Airline",
    "Source",
    "Destination"
]

NUMERICAL_FEATURES = [
    "Total_Stops",
    "Journey_Day",
    "Journey_Month",
    "Dep_Hour",
    "Dep_Minute",
    "Arrival_Hour",
    "Arrival_Minute",
    "Duration_Hours",
    "Journey_Weekday"
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
TARGET_COLUMN = "Price"


# ==========================================================
# 3. DATA LOADING
# ==========================================================

def load_data():
    """Load the original training dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {DATA_PATH}"
        )

    df = pd.read_excel(DATA_PATH)

    print("\nOriginal Dataset Shape:", df.shape)

    return df


# ==========================================================
# 4. DATA CLEANING AND FEATURE ENGINEERING
# ==========================================================

def preprocess_data(df):
    """Clean data and create model-ready features."""

    data = df.copy()

    # Remove exact duplicate records
    data = data.drop_duplicates()

    # Remove rows with missing Route and Total_Stops
    data = data.dropna(
        subset=["Route", "Total_Stops"]
    )

    # Convert journey date
    journey_date = pd.to_datetime(
        data["Date_of_Journey"],
        dayfirst=True,
        errors="coerce"
    )

    # Extract journey date features
    data["Journey_Day"] = journey_date.dt.day
    data["Journey_Month"] = journey_date.dt.month
    data["Journey_Weekday"] = journey_date.dt.dayofweek

    # Extract departure time
    departure_time = pd.to_datetime(
        data["Dep_Time"],
        format="%H:%M",
        errors="coerce"
    )

    data["Dep_Hour"] = departure_time.dt.hour
    data["Dep_Minute"] = departure_time.dt.minute

    # Extract arrival time
    arrival_time = pd.to_datetime(
        data["Arrival_Time"].str.split().str[0],
        format="%H:%M",
        errors="coerce"
    )

    data["Arrival_Hour"] = arrival_time.dt.hour
    data["Arrival_Minute"] = arrival_time.dt.minute

    # Extract flight duration
    duration_parts = data["Duration"].str.extract(
        r"(?:(\d+)h)?\s*(?:(\d+)m)?"
    )

    hours = pd.to_numeric(
        duration_parts[0], errors="coerce"
    ).fillna(0)

    minutes = pd.to_numeric(
        duration_parts[1], errors="coerce"
    ).fillna(0)

    data["Duration_Hours"] = hours + (minutes / 60)

    # Convert Total_Stops into numerical values
    stops_mapping = {
        "non-stop": 0,
        "1 stop": 1,
        "2 stops": 2,
        "3 stops": 3,
        "4 stops": 4
    }

    data["Total_Stops"] = data["Total_Stops"].map(
        stops_mapping
    )

    # Remove rows where feature extraction failed
    data = data.dropna(
        subset=FEATURE_COLUMNS + [TARGET_COLUMN]
    )

    # Keep only required model features and target
    data = data[FEATURE_COLUMNS + [TARGET_COLUMN]]

    print("Cleaned Dataset Shape:", data.shape)
    print("Remaining Missing Values:", data.isnull().sum().sum())

    return data


# ==========================================================
# 5. PREPROCESSING PIPELINE
# ==========================================================

def create_pipeline():
    """Create preprocessing and XGBoost pipeline."""

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES
            ),
            (
                "numerical",
                "passthrough",
                NUMERICAL_FEATURES
            )
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                XGBRegressor(
                    objective="reg:squarederror",
                    random_state=RANDOM_STATE,
                    n_jobs=1
                )
            )
        ]
    )

    return pipeline


# ==========================================================
# 6. XGBOOST HYPERPARAMETER TUNING
# ==========================================================

def tune_model(pipeline, X_train, y_train):
    """Tune XGBoost parameters using RandomizedSearchCV."""

    param_distributions = {
        "model__n_estimators": [200, 300, 400, 500],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__max_depth": [3, 4, 5, 6, 8],
        "model__min_child_weight": [1, 3, 5, 7],
        "model__gamma": [0, 0.1, 0.3, 0.5, 1],
        "model__subsample": [0.6, 0.7, 0.8, 0.9],
        "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9],
        "model__reg_alpha": [0, 0.01, 0.1, 0.5, 1],
        "model__reg_lambda": [1, 3, 5, 10]
    }

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=20,
        cv=3,
        scoring="r2",
        random_state=RANDOM_STATE,
        n_jobs=2,
        verbose=1
    )

    print("\nStarting XGBoost Hyperparameter Tuning...")

    random_search.fit(X_train, y_train)

    print("\nBest Parameters:")
    print(random_search.best_params_)

    print("\nBest Cross Validation R2:")
    print(random_search.best_score_)

    return random_search.best_estimator_


# ==========================================================
# 7. MODEL EVALUATION
# ==========================================================

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    print("\n========== MODEL EVALUATION ==========")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2   : {r2:.4f}")
    print("======================================")

    return mae, rmse, r2


# ==========================================================
# 8. SAVE FINAL MODEL
# ==========================================================

def save_model(model):
    """Save the final trained pipeline."""

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(model, MODEL_PATH)

    print("\nModel Saved Successfully!")
    print("Model Location:", MODEL_PATH)


# ==========================================================
# 9. MAIN TRAINING WORKFLOW
# ==========================================================

def main():

    print("\nFLIGHT PRICE PREDICTION")
    print("Starting Model Training...\n")

    # Load data
    raw_data = load_data()

    # Clean and engineer features
    clean_data = preprocess_data(raw_data)

    # Separate features and target
    X = clean_data[FEATURE_COLUMNS]
    y = clean_data[TARGET_COLUMN]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    print("\nTraining Shape:", X_train.shape)
    print("Testing Shape:", X_test.shape)

    # Create pipeline
    pipeline = create_pipeline()

    # Tune model on training data
    best_pipeline = tune_model(
        pipeline,
        X_train,
        y_train
    )

    # Evaluate on unseen test data
    evaluate_model(
        best_pipeline,
        X_test,
        y_test
    )

    # Refit selected pipeline on complete cleaned dataset
    print("\nTraining Final Model on Full Dataset...")

    best_pipeline.fit(X, y)

    # Save final pipeline
    save_model(best_pipeline)

    print("\nTraining Workflow Completed Successfully!")


if __name__ == "__main__":
    main()
