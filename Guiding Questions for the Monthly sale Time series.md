Question 7

You're a data analyst at a major consultancy firm. Your client, a large investment group, is interested in understanding the future trends in the U.S. nonstore retail sector, which includes rapidly growing online retail markets. They want to allocate investments wisely by predicting future sales in this sector. Your task is to build models to forecast the monthly sales data for nonstore retailers, using the data from the U.S. Census Bureau. It provides insights into consumer spending trends in the digital era. Helps investment firms make informed decisions on where to allocate funds in the retail sector. Assists nonstore retailers in planning inventory and marketing strategies based on projected sales. Your models could explore various aspects like seasonal trends, the impact of economic indicators, and digital market growth. The forecast can help in identifying potential periods of growth or downturns, guiding both investment strategies and operational decisions for businesses in the nonstore retail sector.

To retrieve the nonstore retailers monthly sales data into Python, follow these steps: Go to the Census website's Current Data section: https://www.census.gov/econ/currentdata/ and select "Advance Monthly Sales for Retail and Food Services" and press Submit.

1. Advanced Monthly Sales for Retail and Food Services
2. start year = 1992, end year = 2023
3. "454: Nonstore Retailers"
4. "Sales - Monthly"
5. U.S. Total
Check the box for "Not Seasonally Adjusted" only.
Click "GET DATA."
Scroll down and you will see a data table along with several file formats that are available for this data: TXT, XLSX_V, XLSX_H, etc. Import the Excel file labeled as XLSX_H (NOT XLSX_VI) directly from the website into Python using the pandas library (Hint: similar(identical to be precise) to how an Excel file is imported into Python from GitHub!).

Organize your data in a data frame whose index is date representing the first day of each months and have Sales store in a column of the data frame. below is I am displaying the first few rows to show how data to be ready for visualization and modeling:

Sales
Date
1992-01-01 6860.0
1992-02-01 6059.0
1992-03-01 6297.0
1992-04-01 6022.0
1992-05-01 5803.0

Now create a graph of data and determine which of the following statements best describes the observable trends?

Stable Seasonality: The data shows consistent seasonal peaks and troughs, indicating strong seasonal patterns with no significant growth trend over time.
Periodic Plateaus: While the overall trend is upward, there are noticeable periods where sales level off before increasing again, possibly reflecting market saturation or adoption phases. 
Sudden Growth Spurts: The data reveals periods of stability interspersed with sudden bursts of growth, which may correspond to external market influences or specific promotional campaigns. 
Linear Progression with Outliers: The sales data indicates a steady, straight-line increase over time, punctuated by occasional and irregular outliers that do not follow a seasonal pattern. 
Exponential Growth with Seasonal Fluctuations: The data illustrates a rapid increase in sales volume over time, compounded by recurring seasonal spikes, suggesting heightened sales activity during specific times of the year.

QUESTION 8

Using the pmdarima library and the following parameter constraints 0 ≤ p, q≤ 4, 0 ≤d, D≤2, 0 ≤ P, Q≤2, and max_order = 14, determine the optimal ARIMA(p,d,q)(P.D,Q) model for the logged data: ARIMA(p= (P= D= Q= ,d= 

Using this champion model, predict the value for December 2023 including the upper and lower 95% prediction interval bounds (undo the logarithm transformation and round it to the second decimal place): 
Lower 95% interval bound = 
Predicted value for December 2023 = 
Upper 95% interval bound = 

Clarification: Note that you do not need to divide data into training and testing sets for this question.

QUESTION 9

Can we conclude that the optimal model identified by the auto_arima function is adequate?

© No, because the absence of significant autocorrelations in the ACF only indicates non-stationarity, not white noise characteristics.
No because the residuals are stationary.
Yes because the residuals are nonstationary.
No, because the ACF shows significant autocorrelations.
Yes, because the ACF shows no significant autocorrelations, indicating that the residuals are white noise, i.e., the residuals form a stationary, uncorrelated sequence.

QUESTION 10

You have decided to use the PyCaret library to determine the best performing model for future forecasts, utilizing a time series cross-validation approach. Previously, a model was built using the entire dataset and the penalized error metric AIC to identify the champion model with the pmd library. PyCaret will be used to build a variety of classical time series and machine learning models. Given that forecasts are typically generated for the next 3 months in this application, you've chosen to use 10-fold cross-validation, with each fold consisting of 3 months. The average MAPE (Mean Absolute Percentage Error) of the 10 folds will be used to select the champion model. Due to the multiplicative pattern in the data, apply a Box-Cox transformation to the target variable (ensure transform_target="box-cox' is included in the setup). For grading purposes and to ensure reproducibility, set the seed to 123 in the setup.

Questions about the champion model:

Champion Model Name: Provide the abbreviated name of the champion model as indicated in the first column of the output from compare_models():
Average MAPE: What is the average MAPE of the champion model based on the 10 folds, each consisting of 3 months? (Multiply the MAPE by 100% and round your answer to the second decimal place.)
Largest MAPE: Out of the 10 MAPEs calculated, which MAPE is the largest - provide that largest MAPE? (Multiply the MAPE by 100% and round your answer to the second decimal place.)
Tuning the Champion Model: Tune the champion model and provide the average MAPE of the tuned champion model based on the 10 folds, each consisting of 3 months. (Multiply the MAPE by 100% and round your answer to the second decimal place.)
Future Prediction: Refit the tuned champion model to the entire dataset and predict the value for December 2023. (Round your answer to the second decimal place.)

Make sure to carefully follow the instructions for each part of the question to ensure accuracy in your results.