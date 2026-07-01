import pandas as pd

# Load the dataset
data = pd.read_csv("dataset/heart.csv")

# Show first 10 rows
print("\nFirst 10 Rows")
print(data.head(10))

# Show dataset shape
print("\nDataset Shape")
print(data.shape)

# Show column names
print("\nColumn Names")
print(data.columns)

# Show information
print("\nDataset Information")
print(data.info())

# Check for missing values
print("\nMissing Values")
print(data.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows")
print(data.duplicated().sum())