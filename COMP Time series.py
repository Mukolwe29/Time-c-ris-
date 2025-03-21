# %%
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


# %% [markdown]
# First we load the dataset from the Github
# 

# %%
# Load the dataset
url = "https://github.com/robertasgabrys/DSO424Fall2023/raw/main/CompetitionData.xlsx"
df = pd.read_excel(url)


# %% [markdown]
# We then checked the dimensions of the dataset

# %%
# Check the dimensions of the DataFrame
dimensions = df.shape
print(dimensions)


# %%
summary_stats = df.describe()
print(summary_stats)


# %% [markdown]
# Check for missing data
# 
# 

# %%
# Check for missing values in the DataFrame
missing_values = df.isnull().sum()
print(missing_values)


# %% [markdown]
# The load column had 8760 missing observations
# 
# 

# %% [markdown]
# Drop the missing values
# 

# %%
# Drop rows with missing 'Load' values
df_cleaned = df.dropna(subset=['Load'])
df_cleaned

# %% [markdown]
# We dropped the  rows with missing values

# %% [markdown]
# Since our data had no datetime column ,we created one,we generated it based on hourly frequency

# %%
# the data doesn't have a datetime column
# Generate a time variable based on hourly frequency
df_cleaned['Time'] = pd.date_range(start="2002-01-01 00:00:00", periods=len(df_cleaned), freq="H")




# %% [markdown]
# We then sorted values by time 
# 

# %%
df_sorted = df_cleaned.sort_values(by='Time')

# Display the sorted DataFrame
print(df_sorted.head())  # Display the first few rows to verify the sorting

# %% [markdown]
# 

# %% [markdown]
# Let's start with the exploratory data analysis (EDA) to identify seasonality patterns in our dataset.

# %% [markdown]
# Graph of the Entire Load Dataset (Annual Seasonality)

# %%
import matplotlib.pyplot as plt
import pandas as pd

# Assuming 'Time' is your time column in df_sorted
# Convert 'Time' column to datetime format
df_sorted['Time'] = pd.to_datetime(df_sorted['Time'])

# Plotting the graph again with the updated 'Time' format
plt.figure(figsize=(10, 6))
plt.plot(df_sorted['Time'], df_sorted['Load'])
plt.xlabel('Time')
plt.ylabel('Load')
plt.title('Load Over Time')
plt.show()



# %% [markdown]
# The regular spikes after 6 motnths signifies trend in our dataset
# 

# %% [markdown]
# For a shorter period,we can slice the dataframe to cover only a few days or weeks

# %%
plt.figure(figsize=(10, 6))
plt.plot(df_sorted['Time'].iloc[:100], df_sorted['Load'].iloc[:100])  # Plotting for the first 100 data points
plt.xlabel('Time')
plt.ylabel('Load')
plt.title('Load Over a Few Days')
plt.show()


# %% [markdown]
# From this graph their increasing trend ,Consistent rise  and fall in electricity demand within 24 hour period indicating peak hours and off peak hours shows daily seasonality
# 

# %% [markdown]
# For a few weeks:

# %%
plt.figure(figsize=(10, 6))
plt.plot(df_sorted['Time'].iloc[:1000], df_sorted['Load'].iloc[:1000])  # Plotting for the first 1000 data points
plt.xlabel('Time')
plt.ylabel('Load')
plt.title('Load Over a Few Weeks')
plt.show()

# %% [markdown]
# The consumption of electricity in week days is higher than the weekends 

# %% [markdown]
# To compare Load with Temperature:

# %%
plt.figure(figsize=(10, 6))
plt.scatter(df_sorted['Tavg'], df_sorted['Load'])
plt.xlabel('Temperature (Tavg)')
plt.ylabel('Load')
plt.title('Load vs Temperature')
plt.show()



# %% [markdown]
#  As temperatures increases from 0 to 60 the load decreases ,however as  it increases  from 65, the Load also increases  forming the U-shaped pattern.
#  

# %% [markdown]
# Trend Feature

# %%
df_sorted['Trend'] = range(1, len(df_sorted) + 1)


# %% [markdown]
# Daily Seasonality and Non linear temperature relationships
# 

# %%


# %% [markdown]
# Dont get worried with the warning ,the code stills works 

# %% [markdown]
# Nonlinear Temperature Relationships
# 

# %%
df_sorted['Tavg_Squared'] = df_sorted['Tavg'] ** 2
df_sorted['Tavg_Cubed'] = df_sorted['Tavg'] ** 3
# Add higher orders or other transformations as needed


# %% [markdown]
# These transformations create new features that capture potentially nonlinear relationships between temperature and the target variable. Higher-order transformations like squares and cubes can help capture more complex patterns that might exist in the data, providing the machine learning model with additional information to make predictions. 

# %%


# %% [markdown]
# We created lagged values for the past 6 hours 
# Lag features are crucial in time series forecasting as they capture the historical patterns and dependencies in the data. By creating lag features for the 'Load' variable for the past 6 hours ('Load_Lag_1' to 'Load_Lag_6'), the model can learn from the immediate past data. This allows it to understand how the load has changed over recent hours, which often has a direct impact on future energy consumption patterns. These lag features provide the model with valuable temporal context, enhancing its ability to make accurate predictions based on the recent history of energy load.

# %%
# Create lag features for the past 6 hours (adjust the range as needed)
for i in range(1, 7):
    df_sorted[f'Load_Lag_{i}'] = df_sorted['Load'].shift(i)


# %% [markdown]
# We then calculated the rolling mean of temperature 
# The rolling mean is a method to smooth out fluctuations or noise in time series data, revealing underlying trends. Calculating the rolling mean of 'Tavg' over a 24-hour window ('Tavg_RollingMean') helps capture the average temperature trends over a day. This smoothed representation of temperature variation aids in identifying larger patterns or cycles that could influence energy consumption. By incorporating this feature, the model gains insight into the broader trends in temperature changes, which might correlate with changes in energy demand.
# 

# %%
rolling_window = 24  # 24-hour window
df_sorted['Tavg_RollingMean'] = df_sorted['Tavg'].rolling(window=rolling_window).mean()


# %% [markdown]
# Extract date and time components (hour, day, month, etc.) from the 'Time' column to include as additional features.

# %% [markdown]
# Extracting date and time components such as hour, day, month, etc., from the 'Time' column allows the model to understand and incorporate temporal patterns and seasonality in energy consumption. For instance, different times of the day or specific months might exhibit varying energy demand patterns. By breaking down time into its components, the model can discern and learn from these patterns, thus improving its ability to predict load demand at different times of the day or year.

# %%
df_sorted['Hour'] = df_sorted['Time'].dt.hour
df_sorted['Day'] = df_sorted['Time'].dt.day
df_sorted['Month'] = df_sorted['Time'].dt.month


# %% [markdown]
# In summary, these feature engineering steps enrich the dataset by providing the model with relevant historical context, smoothed representations of temperature trends, and temporal components. This empowers the model to capture intricate patterns and dependencies in the data, leading to more accurate and robust long-term energy load forecasts.
# 

# %% [markdown]
# Now lets Check the relationships of our variables
# 

# %%
# Calculate correlations
correlation_matrix = df_sorted[['Tavg', 'Tmax', 'Tmin', 'Load']].corr()
correlation_matrix

# Visualize correlation matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix (Temperature vs Load)')
plt.show()


# %% [markdown]
# These correlations highlight some degree of relationship between temperature variables (Tavg, Tmax, Tmin) and the energy load. However, the correlations are not very high, signifying that other factors beyond temperature might also play significant roles in influencing energy demand.

# %%
# Lagged Temperature Features
for i in range(1, 7):
    df_sorted[f'Temp_Lag_{i}'] = df_sorted['Tavg'].shift(i)  # Assuming 'Tavg' is the average temperature

# Rolling Mean of Temperature
rolling_window = 24  # 24-hour window
df_sorted['Tavg_RollingMean'] = df_sorted['Tavg'].rolling(window=rolling_window).mean()

# Temperature Range
df_sorted['Temp_Range'] = df_sorted['Tmax'] - df_sorted['Tmin']


# %% [markdown]
# Columns like 'Temp_Lag_1' to 'Temp_Lag_6' now store temperature values from 1 to 6 hours ago.
# 'Tavg_RollingMean' contains the rolling average of the average temperature over a 24-hour window.
# 'Temp_Range' column displays the range of temperature variations within each hour (Tmax - Tmin).
# These engineered features capture historical temperature trends, smoothed temperature patterns, and temperature variations over a given period. They aim to provide the model with more information to potentially improve the accuracy of load forecasting by considering different aspects of temperature behavior and their potential impact on energy demand.

# %% [markdown]
# Model building

# %%
# Selecting features and target variable
features = ['Tavg', 'Tmed', 'Tmax', 'Tmin']
target = 'Load'

# %% [markdown]
# These selected features will be used to train a machine learning model to predict the 'Load' column (electricity load) based on the given temperature variables ('Tavg', 'Tmed', 'Tmax', 'Tmin').

# %%
# Split the data into training and testing sets
train_size = int(len(df_sorted) * 0.8)
train, test = df_sorted[:train_size], df_sorted[train_size:]

# %% [markdown]
# This is an ensemble learning method that constructs multiple decision trees during training and outputs the average prediction of the individual trees. It's effective for regression tasks like predicting continuous values.

# %%
# Model Training
model = RandomForestRegressor(random_state=42)
model.fit(train[features], train[target])

# %%
# Model Prediction
predictions = model.predict(test[features])
predictions

# %%
# Model Evaluation
mae = mean_absolute_error(test[target], predictions)
mape = mean_absolute_percentage_error(test[target], predictions)

# %%
# Print metrics
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Absolute Percentage Error (MAPE): {mape}')

# %% [markdown]
# Weather Influence: Leverage temperature data's impact on electricity consumption for improved forecasts.

# %%
# Check for missing values
print(df_sorted.isnull().sum())

# Handle missing values (replace NaNs with mean for simplicity)
df_sorted.fillna(df_sorted.mean(), inplace=True)

# Define features and target variable
features = ['Tavg', 'Tmax', 'Tmin', 'Temp_Lag_1', 'Temp_Lag_2', 'Load_Lag_1', 'Load_Lag_2']
target = 'Load'

# Splitting the data into training and testing sets
train_size = int(len(df_sorted) * 0.8)
train, test = df_sorted[:train_size], df_sorted[train_size:]

# Model Training
model = RandomForestRegressor(random_state=42)
model.fit(train[features], train[target])

# Model Prediction and Evaluation
prediction2 = model.predict(test[features])

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

mae = mean_absolute_error(test[target], predictions)
mape = mean_absolute_percentage_error(test[target], predictions)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Absolute Percentage Error (MAPE): {mape}')


# %% [markdown]
# Exporting the Predictions

# %%
import os

# Assuming 'predictions' contains the forecasted load values
# Create a DataFrame for predictions with corresponding timestamps
submission_df = pd.DataFrame({'Time': test['Time'], 'Predicted_Load': prediction2})

# Define the file path to the Downloads directory
prediction2_path = os.path.join(os.path.expanduser("~"), "Desktop", "CompetitionSubmissionTemplate.xlsx")

# Save the predictions to the Downloads directory
submission_df.to_excel(prediction2_path, index=False)



