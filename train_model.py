# ==========================================
# Cardiovascular Disease Prediction Project
# Module 3 - Training the First Model
# ==========================================

# Import Libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load Dataset
data = pd.read_csv("dataset/heart.csv")

# Display first five rows
print("\nFirst Five Rows")
print(data.head())

# ==========================================
# Selecting Features (X)
# ==========================================

X = data.drop("Heart Disease", axis=1)

# ==========================================
# Selecting Target (y)
# ==========================================

y = data["Heart Disease"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)
# ==========================================
# Split Dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)
# ==========================================
# Train Logistic Regression Model
# ==========================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("\nModel training completed successfully!")
# ==========================================
# Make Predictions
# ==========================================

predictions = model.predict(X_test)

print("\nPredictions")
print(predictions)
# ==========================================
# Evaluate Model
# ==========================================

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy")

print(f"{accuracy * 100:.2f}%")