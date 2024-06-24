import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load CSV file into a pandas DataFrame
file_path = 'path/to/dataset.csv'
df = pd.read_csv(file_path)

# Separate features and target column
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]   # Target column

# Initialize Random Forest Classifier
model = RandomForestClassifier()

# Fit the model to compute feature importances
model.fit(X, y)

# Get feature importances
importances = model.feature_importances_

# Create a DataFrame to store feature importances
feature_importances_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})

# Sort features by importance in descending order
feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

# Select top k important features (adjust k as needed)
k = 100  # Select top 100 features
top_features = feature_importances_df.head(k)['Feature'].tolist()

# Filter the DataFrame to keep only the top features and the target column
selected_df = df[top_features + ['Target']]

# Save the selected features DataFrame to a new CSV file
output_file_path = 'path/to/dataset.csv'
selected_df.to_csv(output_file_path, index=False)