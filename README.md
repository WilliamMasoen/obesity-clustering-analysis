# Obesity Levels Clustering Analysis

This project aims to analyze the **Obesity Levels Based on Eating Habits and Physical Condition** dataset using various data preprocessing techniques and clustering algorithms. After applying several clustering methods, DBSCAN was chosen as the best performer for this dataset.

## Overview

- **Dataset**: The dataset contains features related to obesity levels based on eating habits and physical conditions.
- **Clustering Algorithms**: Multiple clustering algorithms were tested, including k-means and DBSCAN.
- **Key Steps**:
  1. **Data Preprocessing**:
     - **Handling Missing Values**: Checked for missing values and unusual data.
     - **Categorical Encoding**: Applied one-hot encoding to categorical variables.
     - **Feature Scaling**: Used StandardScaler for scaling numerical features.
     - **Feature Reduction**: Applied PCA to reduce the dimensionality and retain 95% of variance.
  
  2. **Clustering**:
     - Multiple clustering algorithms were explored, and **DBSCAN** was found to perform the best for this dataset.
  
  3. **Evaluation**:
     - Used **Silhouette Score**, **Adjusted Rand Index (ARI)**, and **Normalized Mutual Information (NMI)** to evaluate the clustering performance.
  
  4. **Visualization**:
     - Visualized the clusters in 2D using **t-SNE**.

## Data Preprocessing

- **Data Cleaning**: Ensured no missing values or unusual data points.
- **Categorical Encoding**: Converted categorical features into numeric values using one-hot encoding.
- **Feature Scaling**: Standardized the data using `StandardScaler` to normalize the features.
- **PCA**: Reduced the dimensionality of the data to improve clustering performance, retaining 95% of the variance.

## Clustering Algorithm

- Applied **DBSCAN** clustering due to its ability to find arbitrarily shaped clusters and handle noise in the data.
- Compared DBSCAN with other clustering algorithms, and found it to perform the best based on silhouette score and other evaluation metrics.

## Evaluation Metrics

- **Silhouette Score**: Measured the quality of clusters by assessing how similar points are within their own cluster versus other clusters.
- **Adjusted Rand Index (ARI)**: Compared the clustering results to the actual labels.
- **Normalized Mutual Information (NMI)**: Measured the agreement between the clustering results and true labels.

## Results

- **DBSCAN** performed the best, showing meaningful clusters that were well-separated in the data.
- Visualizations with t-SNE showed clear cluster separation, validating the effectiveness of DBSCAN for this dataset.

## Requirements

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `ucimlrepo`

## Running the Code

1. Install the required libraries.
2. Fetch the dataset using `fetch_ucirepo`.
3. Run the Python script to preprocess the data, apply clustering, and visualize the results.

## Report

A detailed **report file** is included in the project that explains the methodology, data preprocessing techniques, clustering algorithms used, evaluation metrics, and the results.

## Conclusion

The project demonstrates the use of clustering algorithms on a real-world dataset with a thorough data preprocessing pipeline, leading to a successful clustering solution using DBSCAN.
