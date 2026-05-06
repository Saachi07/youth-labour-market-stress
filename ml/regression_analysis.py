import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 1. Load Data
df = pd.read_csv('../research_data_monthly.csv')

# 2. Define Features (X) and Target (y)
# We convert categorical 'Province' into numerical dummy variables
X = pd.get_dummies(df[['Province', 'Participation_Rate', 'Part_Time_Share', 'FT_Wage_Premium', 'Stress_Index', 'Post2020']], drop_first=True)
y = df['Gender_Wage_Gap']

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate
predictions = model.predict(X_test)
r2 = r2_score(y_test, predictions)
print(f"R-squared Score: {r2:.2f} (closer to 1.0 is better)")

# 6. Feature Importance Visualization
# This tells us WHICH economic factors most heavily influence the wage gap
importances = model.feature_importances_
feature_names = X.columns
feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_df.head(10))
plt.title('Top 10 Predictors of the Gender Wage Gap')
plt.tight_layout()
plt.show()