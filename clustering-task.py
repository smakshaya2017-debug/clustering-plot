import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. LOAD YOUR KAGGLE DATASET
# ==========================================
file_path = r"C:\Users\Lenovo\Downloads\archive (3)\housing.csv"
df = pd.read_csv(file_path)

print("--- Starting K-Means Clustering & Plotting ---")

# Drop missing values to keep data clean
df = df.dropna()

# ==========================================
# 2. PREPARE DATA FOR K-MEANS
# ==========================================
# We will cluster based on location (Latitude and Longitude)
X_cluster = df[['longitude', 'latitude']]

# Scale the geographic coordinates for better clustering performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# ==========================================
# 3. RUN K-MEANS CLUSTERING
# ==========================================
# We will group the California houses into 5 geographic zones (clusters)
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
df['location_cluster'] = kmeans.fit_predict(X_scaled)

print(f"Successfully created {num_clusters} location clusters!")

# ==========================================
# 4. PLOTTING THE CLUSTERS (VISUALIZATION)
# ==========================================
plt.figure(figsize=(10, 8))

# Create a scatter plot representing California's geography colored by cluster
sns.scatterplot(
    data=df, 
    x='longitude', 
    y='latitude', 
    hue='location_cluster', 
    palette='viridis', 
    alpha=0.6, 
    edgecolor=None
)

plt.title('K-Means Clustering of California Housing Locations', fontsize=14)
plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)
plt.legend(title='Cluster ID')
plt.grid(True, linestyle='--', alpha=0.5)

print("Displaying the plot window now...")
# Show the interactive plot window
plt.show()