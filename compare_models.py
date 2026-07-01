import pandas as pd

# Split Dataset
from sklearn.model_selection import train_test_split

# Machine Learning Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Evaluation Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
data = pd.read_csv("dataset/heart.csv")
X = data.drop("Heart Disease", axis=1)

y = data["Heart Disease"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
models = {

    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Decision Tree": DecisionTreeClassifier(random_state=42),

    "Random Forest": RandomForestClassifier(random_state=42),

    "KNN": KNeighborsClassifier(),

    "SVM": SVC(),

    "Naive Bayes": GaussianNB()

}
results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    precision = precision_score(
        y_test,
        prediction,
        pos_label="Presence"
    )

    recall = recall_score(
        y_test,
        prediction,
        pos_label="Presence"
    )

    f1 = f1_score(
        y_test,
        prediction,
        pos_label="Presence"
    )

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1
    ])

    print("="*50)
    print(name)
    print("="*50)

    print("Accuracy :", round(accuracy*100,2), "%")
    print("Precision:", round(precision*100,2), "%")
    print("Recall   :", round(recall*100,2), "%")
    print("F1 Score :", round(f1*100,2), "%")

    print("\nConfusion Matrix")

    print(confusion_matrix(y_test, prediction))

    print()
    result_df = pd.DataFrame(
    results,
    columns=[
        "Algorithm",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

print("\n")
print("="*70)
print("MODEL COMPARISON")
print("="*70)

print(result_df)
