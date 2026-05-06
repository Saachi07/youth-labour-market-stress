import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# 1. Load Data
df = pd.read_csv('../research_data_monthly.csv')

# 2. Select Features for Anomaly Detection
# We look at core economic health indicators
anomaly_features = ['Unemployment_Rate', 'CPI_Index', 'Stress_Index', 'Real_Wage']
X_anomaly = df[anomaly_features]

# 3. Scale Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_anomaly)

# 4. Train Isolation Forest
# contamination=0.05 means we assume roughly 5% of our data represents extreme anomalies/shocks
iso_forest = IsolationForest(contamination=0.05, random_state=42)
df['Is_Anomaly'] = iso_forest.fit_predict(X_scaled)

# The model outputs -1 for anomalies and 1 for normal data. Let's map this to True/False
df['Is_Anomaly'] = df['Is_Anomaly'].apply(lambda x: True if x == -1 else False)

# 5. Review the Anomalies
anomalies = df[df['Is_Anomaly'] == True]
print(f"Detected {len(anomalies)} economic anomalies.")
print("\nSample of anomalous events:")
print(anomalies[['YearMonth', 'Province', 'Unemployment_Rate', 'Stress_Index']].head())

# 6. Visualize Anomalies on a timeline (using Unemployment Rate as an example)
plt.figure(figsize=(12, 6))
# Plot normal data
plt.scatter(df[df['Is_Anomaly'] == False]['YearMonth'], df[df['Is_Anomaly'] == False]['Unemployment_Rate'], c='blue', label='Normal', alpha=0.2, s=10)
# Plot anomalies
plt.scatter(anomalies['YearMonth'], anomalies['Unemployment_Rate'], c='red', label='Anomaly', s=30)

plt.title('Anomaly Detection: Flagged Economic Shocks (Unemployment Rate)')
plt.xticks(df['YearMonth'][::150], rotation=45) # Show fewer x-axis labels for readability
plt.legend()
plt.tight_layout()
plt.show()