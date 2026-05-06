import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Load Data
df = pd.read_csv('../research_data_monthly.csv')

# 2. Select Features for Clustering
# We want to cluster based on employment dynamics
cluster_features = ['FT_Employment', 'PT_Employment', 'Part_Time_Share']
X_cluster = df[cluster_features]

# 3. Scale the Data (Crucial for K-Means so large numbers don't overpower percentages)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# 4. Apply K-Means Clustering
# Let's assume we want to find 3 distinct economic personas/profiles
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster_Profile'] = kmeans.fit_predict(X_scaled)

# 5. Analyze the Clusters
print("Average values for each identified cluster:")
cluster_summary = df.groupby('Cluster_Profile')[cluster_features].mean()
print(cluster_summary)

# 6. Visualize the Clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='FT_Employment', y='Part_Time_Share', hue='Cluster_Profile', palette='viridis', alpha=0.6)
plt.title('Economic Segments: Full-Time Employment vs. Part-Time Share')
plt.tight_layout()
plt.show()