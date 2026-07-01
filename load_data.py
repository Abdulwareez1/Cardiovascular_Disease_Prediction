import pandas as pd

# Load dataset
data = pd.read_csv("dataset/heart.csv")

print(data.head())

print()

print(data.info())

print()

print(data.describe())