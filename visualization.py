import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("dataset/heart.csv")

plt.hist(data["age"])

plt.title("Age Distribution")

plt.xlabel("Age")

plt.ylabel("Frequency")

plt.show()