# %%
import pandas as pd

# URL of the Excel file from Census Bureau
url = 'https://www.census.gov/econ_export/?format=xls&mode=report&default=false&errormode=Dep&charttype=&chartmode=&chartadjn=&submit=GET+DATA&program=MARTS&startYear=1992&endYear=2023&categories%5B0%5D=454&dataType=SM&geoLevel=US&adjusted=false&notAdjusted=true&errorData=false'

# Read the Excel file into a pandas DataFrame
df = pd.read_excel(url, skiprows=7)  # Skip the initial 5 rows of metadata




# %% [markdown]
# Print the first first few rows 

# %%
# Display the first few rows of the DataFrame
print(df.head())



# %% [markdown]
# Let us check the dimension of the dataframe 

# %%
# Check the dimensions of the DataFrame
dimensions = df.shape
print(dimensions)

# %% [markdown]
# The dataset has 32 rows and 13 columns 
# 

# %% [markdown]
# Check for Summarry Statistics
# 

# %%
summary_stats = df.describe()
print(summary_stats)


# %% [markdown]
# Let us check for any missing data in our dataset
# 

# %%
# Check for missing values in the DataFrame
missing_values = df.isnull().sum()
print(missing_values)


# %% [markdown]
# There are two missing values in the dataset ,November and December have one each

# %% [markdown]
# Fill the missing values 

# %%
# Fill missing values by forward-filling the missing values
df.fillna(method='ffill', inplace=True)
print(df.isnull().sum())

# %% [markdown]
#  Lets generate a monthly frequency using the first day of each month.

# %%
import pandas as pd

# Assuming your DataFrame is named 'df' and has columns like 'Year', 'Jan', 'Feb', ..., 'Dec'

# Create a list of month names for reference
months = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
]

# Initialize an empty list to hold the data
monthly_sales = []

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    year = str(int(row['Year']))  # Convert year to integer, then to string
    # Iterate through each month to retrieve the sales data
    for month in months:
        # Create a datetime object for the first day of each month
        date = pd.to_datetime(f"{year}-{month}-01")
        # Retrieve sales data for the corresponding month and year
        sales = row[month]
        # Append the date and sales data to the list
        monthly_sales.append({'Date': date, 'Sales': sales})

# Create a new DataFrame from the list of dictionaries
monthly_sales_df = pd.DataFrame(monthly_sales)

# Set the 'Date' column as the index
monthly_sales_df.set_index('Date', inplace=True)

# Displaying the first few rows as requested
print(monthly_sales_df.head())


# %% [markdown]
# Now we create a graph of the monthly data to answer the questions asked
# 

# %% [markdown]
# From the graph there is a rapid and continuous increase in sales over time, accompanied by recurring seasonal spikes. This pattern suggests sustained exponential growth.
# 
# And so Option number 5 is correct Exponential Growth with Seasonal Fluctuations: The data illustrates a rapid increase in sales volume over time, compounded by recurring seasonal spikes, suggesting heightened sales activity during specific times of the year.

# %% [markdown]
# *QUESTION 8*
# 

# %% [markdown]
# Lets get the logged data for our monthly sales data

# %%
import numpy as np

#  'monthly_sales_df' is our DataFrame with 'Sals data

# Create a logged version of the sales data
logged_sales_data = np.log(monthly_sales_df['Sales'])

# Display the first few rows of the logged sales data
print(logged_sales_data.head())

# Now, use the logged sales data with auto_arima to find the optimal ARIMA model
# Insert the code for auto_arima here (as shown in the previous example)


# %% [markdown]
# To determine the optimal ARIMA model for monthly sales data using the pmdarima library and the specified parameter constraints, we can use the auto_arima function. This function will help in automatically selecting the best ARIMA model based on the provided constraints and criteria.

# %% [markdown]
# Installing required pmdarima library

# %%
!pip install pmdarima


# %% [markdown]
# With the logged sales data ready, we can proceed to find the optimal ARIMA model using the auto_arima function from the pmdarima library

# %%
from pmdarima import auto_arima


# Define the parameter constraints and use auto_arima to find the optimal ARIMA model
champion_model = auto_arima(logged_sales_data,
                            start_p=0, max_p=4,
                            start_q=0, max_q=4,
                            d=2, max_d=2,  # Change range(0, 3) to an integer (0, 1, or 2)
                            start_P=0, max_P=2,
                            start_Q=0, max_Q=2,
                            D=2, max_D=2,  # Change range(0, 3) to an integer (0, 1, or 2)
                            max_order=14,
                            seasonal=True,
                            suppress_warnings=True,
                            stepwise=True)

# Get the summary of the optimal model
print(champion_model.summary())


# %% [markdown]
# Optimal ARIMA model for the logged data is ARIMA(4,2,0)

# %%
# Make predictions for December 2023 ( last point in our dataset is October 2023)
n_periods = 2  # Number of periods to forecast (December 2023)

# Use the SARIMAX model to forecast the next n_periods
forecast, conf_int = stepwise_model.predict(n_periods=n_periods, return_conf_int=True)

# Extract the predicted value and confidence intervals for December 2023
pred_value = forecast[0]  # Predicted value for December 2023
lower_bound = conf_int[0, 0]  # Lower 95% interval bound for December 2023
upper_bound = conf_int[0, 1]  # Upper 95% interval bound for December 2023

# Undo the logarithm transformation
pred_value = np.exp(pred_value)
lower_bound = np.exp(lower_bound)
upper_bound = np.exp(upper_bound)

# Round the values to the second decimal place
pred_value = round(pred_value, 2)
lower_bound = round(lower_bound, 2)
upper_bound = round(upper_bound, 2)

print(f"Lower 95% interval bound: {lower_bound}")
print(f"Predicted value for December 2023: {pred_value}")
print(f"Upper 95% interval bound: {upper_bound}")


# %% [markdown]
# These values represent the forecasted sales for December 2023, with a 95% confidence interval suggesting that the actual value is expected to fall within this range.
# Lower 95% interval bound: 107981.59
# Predicted value for December 2023: 139030.07
# Upper 95% interval bound: 179006.06

# %% [markdown]
# *Question 9*
# 

# %% [markdown]
# Can we conclude that the optimal model identified by the auto_arima function is adequate?
# 
# 
# "Yes, because the ACF shows no significant autocorrelations, indicating that the residuals are white noise, i.e., the residuals form a stationary, uncorrelated sequence" aligns with the assessment of model adequacy.

# %% [markdown]
# *QUESTION 10*
# 

# %% [markdown]
# This task involves using the PyCaret library for time series forecasting and evaluating the champion model based on certain criteria

# %% [markdown]
# Step 1: Setup PyCaret Environment

# %%
pip install pycaret

# %% [markdown]
# Setting up the Environment and Importing Libraries

# %%
from pycaret.regression import *


# %% [markdown]
# Prepare  data for analysis and set up the PyCaret environment:

# %%
exp = setup(monthly_sales_df, session_id=123, transform_target='box-cox')


# %% [markdown]
#  Model Selection and Evaluation 

# %%
best_model = compare_models(sort='MAPE', fold=10, round=2)


# %% [markdown]
# Create the Best Model

# %%
final_model = create_model(best_model)


# %% [markdown]
# Model Tuning

# %% [markdown]
#  Future Prediction
# Refit the tuned champion model to the entire dataset and predict the value for December 2023:

# %%
# since 'Date' column exists and is in datetime format
new_data = monthly_sales_df[monthly_sales_df.index.month == 12]
print(new_data)


# %%
# Predict for December 2023
predictions = predict_model(final_model, data=new_data)
predictions['prediction_label'] = predictions['prediction_label'].round(2)
predictions.tail(1)


