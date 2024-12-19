from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

# ----------------------------------------------------------------------------------
# LOADING DATA STARTS HERE
# Fetch dataset -- Obesity Levels Based on Eating Habits and Physical Condition
dataset = fetch_ucirepo(id=544)

# Load the data and combine features and target into variable "data"
data = pd.DataFrame(dataset.data.features, columns=dataset.feature_names)
data["Target"] = dataset.data.targets  # Add the target as a new column
# ----------------------------------------------------------------------------------



# ----------------------------------------------------------------------------------
# DATA PREPROCESSING STARTS HERE
# 1. Data cleaning
# Check if each of the column has missing values
print("Missing values:")
print(data.isnull().sum())
print("------------------------------------------------------------------------------")

# Check for unusual values
print("Check for unusual values")
for column in data.columns:
    unique_values = data[column].unique()
    print("Column:", column, unique_values)
print("------------------------------------------------------------------------------")
# Data cleaning process not needed as there are no missing and unusual values


# 2. Encode categorical values | e.g Gender
# Separate to features and target
x = dataset.data.features 
y = dataset.data.targets

# Encode process
categorical_columns = x.select_dtypes(include = ["object"]).columns
x_encoded = pd.get_dummies(x, columns = categorical_columns, drop_first = True)
print("Shape after encoding:", x_encoded.shape)


# 3. Feature scaling
scaler = StandardScaler()
x_scaled = pd.DataFrame(scaler.fit_transform(x_encoded), columns = x_encoded.columns)
print("Shape after scaling:", x_scaled.shape)


# 4. Feature reduction using PCA technique
# Initialize PCA
pca = PCA(n_components = 0.95)  # Select 95% of variance
x_reduced = pca.fit_transform(x_scaled)

# Plot explained variance ratio
explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize = (8, 5))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker = 'o', linestyle = '--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA - Explained Variance')
plt.grid()
plt.show()

print(f"Number of components that explain 95% of the variance: {x_reduced.shape[1]}")
print("Shape after PCA:", x_reduced.shape)
print("------------------------------------------------------------------------------")
# ----------------------------------------------------------------------------------



# ----------------------------------------------------------------------------------
# CLUSTERING PROCESS STARTS HERE
# Apply DBSCAN Clustering
dbscan = DBSCAN(eps = 0.7, min_samples = 6)
dbscan_clusters = dbscan.fit_predict(x_reduced)

# Evaluate clustering performance using Silhouette Score (excluding noise points)
# Exclude noise points (labeled as -1) for silhouette score calculation
mask = dbscan_clusters != -1  # Mask to filter out noise
if len(set(dbscan_clusters[mask])) > 1:  # Silhouette requires at least 2 clusters
    silhouette_avg = silhouette_score(x_reduced[mask], dbscan_clusters[mask])
    print(f"Silhouette Score (excluding noise): {silhouette_avg}")
else:
    print("Not enough clusters for Silhouette Score calculation (excluding noise).")

# Visualize the clusters using t-SNE for better 2D separation
tsne = TSNE(n_components=2, random_state=42)
x_tsne = tsne.fit_transform(x_reduced)

plt.figure(figsize=(8, 5))
plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=dbscan_clusters, cmap='viridis', marker='o', edgecolor='k', s=50)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title(f'Clusters Visualized with t-SNE (DBSCAN Clustering)')
plt.colorbar(label='Cluster')
plt.grid()
plt.show()

# Compare clusters to true labels (excluding noise points)
# Flatten y to make it 1D
y = y.values.ravel()

ari_score = adjusted_rand_score(y[mask], dbscan_clusters[mask])
nmi_score = normalized_mutual_info_score(y[mask], dbscan_clusters[mask])

print("Adjusted Rand Index (ARI, excluding noise):", ari_score)
print("Normalized Mutual Information (NMI, excluding noise):", nmi_score)
