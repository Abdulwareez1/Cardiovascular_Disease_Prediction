import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load Dataset
data = pd.read_csv("dataset/heart.csv")

# Features and Target
X = data.drop("Heart Disease", axis=1)
y = data["Heart Disease"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save Model
joblib.dump(model, "models/heart_disease_model.pkl")

print("Model saved successfully.")