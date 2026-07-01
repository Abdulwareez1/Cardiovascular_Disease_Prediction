import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("dataset/heart.csv")

# Display first five rows
print(data.head())

# Gender Distribution
plt.figure(figsize=(6,5))

gender = data["Sex"].value_counts()

plt.bar(gender.index.astype(str), gender.values)

plt.title("Gender Distribution")
plt.xlabel("Sex (0 = Female, 1 = Male)")
plt.ylabel("Number of Patients")

plt.show()
# Heart Disease Distribution
plt.figure(figsize=(6,5))

disease = data["Heart Disease"].value_counts()

plt.pie(
    disease,
    labels=["No Disease", "Heart Disease"],
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Heart Disease Distribution")

plt.show()
import seaborn as sns

plt.figure(figsize=(12,8))

sns.heatmap(
    data.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Feature Correlation")

plt.show()