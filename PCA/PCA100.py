import pandas as pd
from sklearn.decomposition import PCA

# Load CSV files into pandas DataFrames
file1 = 'path/to/dataset.csv'
df1 = pd.read_csv(file1)

# Separate features and target column
X1 = df1.iloc[:, :-1]  # Features for file1
y1 = df1.iloc[:, -1]   # Target column for file1

# Initialize PCA with desired number of components
n_components = 100  # Choose the number of components you want to keep
pca = PCA(n_components=n_components)

# Apply PCA to reduce dimensionality
X1_reduced = pca.fit_transform(X1)

# Create new DataFrames with reduced features and target column
df1_reduced = pd.DataFrame(data=X1_reduced, columns=[f'feature_{i}' for i in range(n_components)])
df1_reduced['target'] = y1

# Save the reduced features and target column into new CSV files
output_file1 = 'path/to/dataset.csv'

df1_reduced.to_csv(output_file1, index=False)