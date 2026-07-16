from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("models/heart_disease_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

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

    except Exception as e:
        return f"{e}<br><br>{request.form}"

    # -------------------------------------
    # Input Validation
    # -------------------------------------

    if age < 1 or age > 120:
        return render_template(
            "index.html",
            error="Age must be between 1 and 120 years."
        )

    if bp < 50 or bp > 250:
        return render_template(
            "index.html",
            error="Resting Blood Pressure must be between 50 and 250 mmHg."
        )

    if cholesterol < 100 or cholesterol > 700:
        return render_template(
            "index.html",
            error="Cholesterol must be between 100 and 700 mg/dL."
        )

    if max_hr < 60 or max_hr > 250:
        return render_template(
            "index.html",
            error="Maximum Heart Rate must be between 60 and 250 bpm."
        )

    if st < 0 or st > 10:
        return render_template(
            "index.html",
            error="ST Depression must be between 0 and 10."
        )

    # -------------------------------------
    # Prepare Data for Prediction
    # -------------------------------------

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

    # -------------------------------------
    # Prediction
    # -------------------------------------

    prediction = model.predict(patient)[0]

    probability = model.predict_proba(patient)[0]

    confidence = round(max(probability) * 100, 2)

    # -------------------------------------
    # Recommendation
    # -------------------------------------

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

    # -------------------------------------
    # Display Result
    # -------------------------------------

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        risk=risk,
        recommendation=recommendation
    )


# -------------------------------------
# Run Flask App
# -------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )