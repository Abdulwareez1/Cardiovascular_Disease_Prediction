from flask import Flask, render_template, request
import pandas as pd
import joblib

# -------------------------------------
# Create Flask App
# -------------------------------------

app = Flask(__name__)

# -------------------------------------
# Load Trained Model
# -------------------------------------

model = joblib.load("models/heart_disease_model.pkl")


# -------------------------------------
# Home Page
# -------------------------------------

@app.route("/")
def home():

    return render_template("index.html")


# -------------------------------------
# Prediction Page
# -------------------------------------

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

    patient = pd.DataFrame(
        [[
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
        ]],
        columns=[
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
        ]
    )

    prediction = model.predict(patient)[0]

    probability = model.predict_proba(patient)[0]

    confidence = round(max(probability) * 100, 2)

    if prediction == "Presence":

        risk = "HIGH"

        recommendation = (
            "The patient appears to be at high risk of cardiovascular disease. "
            "Please consult a qualified healthcare professional for a complete evaluation."
        )

    else:

        risk = "LOW"

        recommendation = (
            "The patient appears to have a low predicted risk of cardiovascular disease "
            "based on the information provided. Continue maintaining a healthy lifestyle "
            "and attend regular medical check-ups."
        )

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        risk=risk,
        recommendation=recommendation
    )


# -------------------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )