# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
#file path to be changed
df = pd.read_csv('C:\\Users\\RVCFF112\\Downloads\\car_dataset_modified.csv')

# Check if the dataset is loaded correctly
print(df.head())  # Print the first few rows of the dataset to verify

# Preprocessing
# Check for missing values before dropping them
print(df.isnull().sum())  # Print the count of missing values for each column

# Drop any rows with missing values
df.dropna(inplace=True)

# Convert categorical variables into numerical using one-hot encoding
df = pd.get_dummies(df, columns=['make', 'fuel-type', 'number-of-doors'])

# Confirm the columns after one-hot encoding
print(df.columns)

# Split the dataset into features and target variable
X = df.drop(columns=['price'])
y = df['price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shapes of training and testing sets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a regression model
reg_model = LinearRegression()
reg_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = reg_model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)
