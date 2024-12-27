'''
Gold price prediction by machine learning
Author: Henry Ha
'''

# Import the necessary libraries
import pandas as pd

#TODO EDA

# Load the dataset
users_data = pd.read_csv('gld_price_data.csv')

# Check the structure of the dataset
print(users_data.info())
print(users_data.describe())
print(users_data.head())

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

# Plot gold prices over time
plt.figure(figsize=(12, 6))
plt.plot(users_data['Date'], users_data['GLD'], label='Gold Price', color='gold')

# Format the x-axis to show more dense date labels
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Set ticks every 3 months
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# Rotate the date labels for better readability
plt.xticks(rotation=45)
plt.title('Gold Price Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Gold Price')
plt.legend()
plt.tight_layout()
plt.show()

# Exclude the 'Date' column before calculating correlations
correlation_matrix = users_data.drop(columns=['Date']).corr()

# Plot the heatmap
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap (Excluding Date)')
plt.show()

#TODO Data preprocessing

# Check for missing values
print(users_data.isnull().sum())

from sklearn.preprocessing import StandardScaler

# Selecting numerical features (excluding 'Date')
features = users_data.drop(columns=['Date'])
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Convert scaled features back to a DataFrame
scaled_data = pd.DataFrame(scaled_features, columns=features.columns)

# Separate features and target
X = scaled_data.drop(columns=['GLD'])
y = scaled_data['GLD']

from sklearn.model_selection import train_test_split

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#TODO Model building and evaluation

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Train Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_lr = lr_model.predict(X_test)
print("Linear Regression R^2 Score:", r2_score(y_test, y_pred_lr))
print("Linear Regression RMSE:", mean_squared_error(y_test, y_pred_lr, squared=False))

from sklearn.ensemble import RandomForestRegressor

# Train Random Forest
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_rf = rf_model.predict(X_test)
print("Random Forest R^2 Score:", r2_score(y_test, y_pred_rf))
print("Random Forest RMSE:", mean_squared_error(y_test, y_pred_rf, squared=False))

from sklearn.ensemble import GradientBoostingRegressor

# Train Gradient Boosting
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_gb = gb_model.predict(X_test)
print("Gradient Boosting R^2 Score:", r2_score(y_test, y_pred_gb))
print("Gradient Boosting RMSE:", mean_squared_error(y_test, y_pred_gb, squared=False))

import joblib
# Save the trained model to a pickle file
joblib.dump(rf_model, 'random_forest_model.pkl')
print("Model saved as 'random_forest_model.pkl'")