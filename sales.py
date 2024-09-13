# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Load the dataset
sales_data = pd.read_csv('sales_data.csv')

# Display the first few rows
sales_data.head()
# Handle missing values by filling with 0 (if applicable)
sales_data.fillna(0, inplace=True)

# Convert Date to datetime type
sales_data['Date'] = pd.to_datetime(sales_data['Date'])

# Create new features from the 'Date' column
sales_data['Year'] = sales_data['Date'].dt.year
sales_data['Month'] = sales_data['Date'].dt.month
sales_data['Day'] = sales_data['Date'].dt.day
sales_data['DayOfWeek'] = sales_data['Date'].dt.dayofweek

# Create lag features (Previous day sales as input)
sales_data['Sales_Lag_1'] = sales_data['Sales'].shift(1)

# Rolling window features (7-day rolling average)
sales_data['Rolling_Mean_7'] = sales_data['Sales'].rolling(window=7).mean()

# Fill any NaN values created by rolling mean with 0
sales_data.fillna(0, inplace=True)

# Check the processed data
sales_data.head()
# Plot sales over time to see trends
plt.figure(figsize=(12,6))
plt.plot(sales_data['Date'], sales_data['Sales'], label='Sales')
plt.title('Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

# Check correlation heatmap to understand relationships between features
plt.figure(figsize=(10, 6))
sns.heatmap(sales_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation between features')
plt.show()

# Define features (X) and target (y)
features = ['Year', 'Month', 'DayOfWeek', 'Sales_Lag_1', 'Rolling_Mean_7', 'IsHoliday']
X = sales_data[features]
y = sales_data['Sales']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Function to evaluate the model
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f'{model_name} Performance:')
    print(f'MAE: {mae:.2f}')
    print(f'RMSE: {rmse:.2f}')
    print(f'R2 Score: {r2:.2f}')
    print('-'*30)

# Evaluate Linear Regression
evaluate_model(y_test, lr_predictions, "Linear Regression")

# Evaluate Random Forest
evaluate_model(y_test, rf_predictions, "Random Forest")

# Visualize Actual vs Predicted for Linear Regression
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Sales', color='blue')
plt.plot(lr_predictions, label='Predicted Sales (Linear Regression)', color='red')
plt.title('Actual vs Predicted Sales (Linear Regression)')
plt.legend()
plt.show()

# Visualize Actual vs Predicted for Random Forest
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Sales', color='blue')
plt.plot(rf_predictions, label='Predicted Sales (Random Forest)', color='green')
plt.title('Actual vs Predicted Sales (Random Forest)')
plt.legend()
plt.show()
