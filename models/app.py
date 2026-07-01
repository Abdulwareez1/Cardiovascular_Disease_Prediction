from flask import Flask, render_template, request
import joblib
import pandas as pd

# Create Flask application
app = Flask(__name__)

# Load the trained model
model = joblib.load("models/heart_disease_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    age = float(request.form["Age"])
    sex = int(request.form["Sex"])
    chest_pain = int(request.form["Chest pain type"])
    bp = float(request.form["BP"])
    cholesterol = float(request.form["Cholesterol"])
    fbs = int(request.form["FBS over 120"])
    ekg = int(request.form["EKG results"])
    max_hr = float(request.form["Max HR"])
    exercise = int(request.form["Exercise angina"])
    st = float(request.form["ST depression"])
    slope = int(request.form["Slope of ST"])
    vessels = int(request.form["Number of vessels fluro"])
    thallium = int(request.form["Thallium"])

    patient = pd.DataFrame([[

        age,
        sex,
        chest_pain,
        bp,
        cholesterol,
        fbs,
        ekg,
        max_hr,
        exercise,
        st,
        slope,
        vessels,
        thallium

    ]], columns=[

        "Age",
        "Sex",
        "Chest pain type",
        "BP",
        "Cholesterol",
        "FBS over 120",
        "EKG results",
        "Max HR",
        "Exercise angina",
        "ST depression",
        "Slope of ST",
        "Number of vessels fluro",
        "Thallium"

    ])

    prediction = model.predict(patient)[0]

    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)