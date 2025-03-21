# Time Series Analysis and Forecasting

This repository contains multiple scripts and notebooks for time series analysis and forecasting using machine learning and statistical methods. The focus is on analyzing energy load and retail sales data, performing feature engineering, and building predictive models.

## Repository Contents

### Notebooks and Scripts

1. **`Competition time series.ipynb`**
   - Implements time series forecasting using `RandomForestRegressor`.
   - Includes feature engineering such as lag features, rolling means, and temporal components.
   - Evaluates model performance using MAE and MAPE metrics.

2. **`Timeseriescompetition.py`**
   - Loads and preprocesses energy load data.
   - Performs exploratory data analysis (EDA) to identify seasonality and trends.
   - Builds a machine learning model for load forecasting using temperature and lagged features.

3. **`TIME series non store retail Monthly sales.ipynb`**
   - Analyzes monthly retail sales data.
   - Implements feature engineering and ARIMA modeling for forecasting.
   - Uses `auto_arima` to identify the optimal ARIMA model and forecasts future sales.

4. **`COMP Time series.py`**
   - Similar to `Timeseriescompetition.py` but includes additional EDA and feature engineering steps.
   - Visualizes relationships between temperature and energy load.
   - Exports predictions to an Excel file.

## Features

- **Data Preprocessing**: Handles missing values, creates datetime columns, and generates lagged features.
- **Exploratory Data Analysis (EDA)**: Visualizes trends, seasonality, and correlations.
- **Feature Engineering**: Includes lag features, rolling means, and nonlinear transformations.
- **Modeling**:
  - Machine Learning: Random Forest for regression tasks.
  - Statistical Models: ARIMA for time series forecasting.
- **Evaluation**: Metrics such as Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE).
- **Export**: Saves predictions to Excel files for submission.

## Requirements

To run the code, install the following Python libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `pmdarima`
- `openpyxl`

Install the required libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn pmdarima openpyxl