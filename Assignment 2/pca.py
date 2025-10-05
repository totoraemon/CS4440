# ------------------------------------------------------------------------------
# AUTHOR: Joseline Ly
# FILENAME: pca.py
# SPECIFICATION: The program performs PCA on the heart_disease_dataset.csv file,
#                removing one feature at a time and calculating the variance of
#                the first principal component (PC1).
# FOR: CS 4440 (Data Mining) - Assignment #2
# TIME SPENT: 2 hours
# -----------------------------------------------------------------------------*/

#importing some Python libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the data
#--> add your Python code here
df = pd.read_csv('heart_disease_dataset.csv')

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

#Get the number of features
#--> add your Python code here
num_features = df.shape[1]

# Run PCA using 9 features, removing one feature at each iteration
variances = []  # List to store variances
features_removed = []  # List to store removed features

for i in range(num_features):
    # Create a new dataset by dropping the i-th feature
    # --> add your Python code here
    reduced_data = np.delete(scaled_data, i, axis=1)

    # Run PCA on the reduced dataset
    # --> add your Python code here
    pca = PCA(n_components=1)
    pca.fit(reduced_data)

    #Store PC1 variance and the feature removed
    #Use pca.explained_variance_ratio_[0] and df_features.columns[i] for that
    # --> add your Python code here
    pc1_variance = pca.explained_variance_ratio_[0]
    feature_removed = df.columns[i]
    
    # Used to ensure that features were removed properly with variances in correct order
    # print(f'Removed feature: {feature_removed}, PC1 Variance: {pc1_variance}')

    # Append the results to the lists
    variances.append(pc1_variance)
    features_removed.append(feature_removed)

# Find the maximum PC1 variance and the corresponding feature
# --> add your Python code here
max_variance = max(variances)
feature_to_remove = features_removed[variances.index(max_variance)]

# Print results
# Use the format: Highest PC1 variance found: ? when removing ?
# --> add your Python code here
print(f'Highest PC1 variance found: {max_variance} when removing {feature_to_remove}')