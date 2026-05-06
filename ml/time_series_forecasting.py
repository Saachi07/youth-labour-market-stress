import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 1. Load Data
df = pd.read_csv('../research_data_monthly.csv')

# 2. Preprocessing: Sort by date to maintain time series integrity
df['Date'] = pd.to_datetime(df['YearMonth'])
df = df.sort_values('Date')

# 3. Feature Engineering: Create a 'Lagged' Unemployment Rate (previous month)
# We group by Province and Age_Group so we don't mix different demographic timelines
df['Lag_1_Unemployment'] = df.groupby(['Province', 'Age_Group'])['Unemployment_Rate'].shift(1)

# Drop the first row of each group since it won't have a previous month to reference
df_clean = df.dropna(subset=['Lag_1_Unemployment']).copy()

# 4. Define Features (X) and Target (y)
features = ['Lag_1_Unemployment', 'Participation_Rate', 'CPI_Index', 'Stress_Index']
X = df_clean[features]
y = df_clean['Unemployment_Rate']

# 5. Split Data (Sequential split is better for time series, but we use train_test_split for simplicity here without shuffling)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 6. Train the Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Evaluate and Visualize
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse:.2f}")

# Plot actual vs predicted for a visual check
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:100], label='Actual Unemployment Rate', marker='o')
plt.plot(predictions[:100], label='Predicted Unemployment Rate', marker='x')
plt.title('Time Series Forecasting: Actual vs Predicted Unemployment (First 100 Test Samples)')
plt.legend()
plt.tight_layout()
plt.show()