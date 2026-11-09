# ✈️ Flight Price Prediction

A Machine Learning project that predicts estimated airline ticket prices based on flight details such as airline, source, destination, journey date, departure time, arrival time, number of stops, and flight duration.

## Project Overview

The objective of this project is to build a regression model that estimates flight ticket prices using historical flight data.

The project includes data inspection, exploratory data analysis, feature engineering, model training, hyperparameter tuning, model evaluation, and a Streamlit web application.

## Features

- Flight price prediction using XGBoost.
- Exploratory Data Analysis (EDA).
- Feature engineering, including date and time features.
- Multiple machine learning model comparisons.
- Hyperparameter tuning using RandomizedSearchCV.
- Cross-validation for model evaluation.
- Automated preprocessing using Scikit-learn Pipeline.
- Interactive Streamlit web application.

## Model Performance

**Final XGBoost Pipeline:**

| Metric | Score |
|---|---:|
| MAE | 1118.57 |
| RMSE | 1847.60 |
| R² Score | 0.8363 |
| 5-Fold CV R² | 0.8433 |

The model achieved an R² score of approximately 83.63% on the holdout test set before final training on the complete dataset.

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Streamlit
- Joblib
- Jupyter Notebook

## Project Structure

```text
Flight_Price_Prediction/
│
├── data/
│   ├── Data_Train.xlsx
│   ├── Test_set.xlsx
│   └── Sample_submission.xlsx
│
├── models/
│   └── flight_price_prediction_pipeline.pkl
│
├── notebooks/
│   └── 01_Data_Inspection.ipynb
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project folder

```bash
cd Flight_Price_Prediction
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your web browser.

## How to Use

1. Select an airline.
2. Choose source and destination cities.
3. Select the journey date.
4. Enter departure and arrival times.
5. Select the number of stops.
6. Enter flight duration.
7. Click **Predict Flight Price**.
8. View the estimated ticket price.

## Important Note

This application provides dataset-based estimated flight prices. It does not fetch real-time airline ticket prices.

## Future Improvements

- Improve model performance through advanced feature engineering.
- Add more machine learning algorithms.
- Deploy the application online.
- Integrate real-time flight fare APIs.

## Author

**Mrinal Dutta**

Data Science | Machine Learning | Python