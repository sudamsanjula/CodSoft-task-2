import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
data_path = input("Enter the path for the dataset: ")
data = pd.read_csv(data_path)

# Step 2: Exploratory Data Analysis (EDA)
print("First few rows of the dataset:")
print(data.head())

print("\nBasic statistics of the dataset:")
print(data.describe())

print("\nChecking for missing values:")
print(data.isnull().sum())

# Visualize the relationships between features and the target variable
sns.pairplot(data)
plt.show()

# Step 3: Data Preprocessing
# Handling missing values (if any)
data = data.dropna()

# Step 4: Feature Engineering
# Let's assume 'Sales' is our target variable and others are features
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Step 5: Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Building and training the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 8: Evaluating the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'\nMean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Step 9: Making predictions
new_data = [[230.1, 37.8, 69.2]]  # Example new data
new_data_scaled = scaler.transform(new_data)
predicted_sales = model.predict(new_data_scaled)
print(f'\nPredicted Sales for new data {new_data}: {predicted_sales[0]}')

# Visualizing the results
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.show()
