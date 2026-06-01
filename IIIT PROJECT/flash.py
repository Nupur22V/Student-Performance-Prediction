from flask import Flask, request, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load the entire trained pipeline (Preprocessing + Model)
model = joblib.load("models/model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form values as strings or numbers based on dataset types
        # Note: If your clean dataset has text categories for these features, 
        # they will pass safely into the pipeline as strings now.
        input_data = {
            'study_hours(per_day)': [request.form['study_hours']],
            'sleep_hours': [request.form['sleep_hours']],
            'attendance(%)': [request.form['attendance']],
            'coding_hours(per_day)': [request.form['coding_hours']],
            'social_media_usage': [request.form['social_media']],
            'participation_in_class': [request.form['participation']]
        }

        # Convert dictionary to DataFrame so the ColumnTransformer knows column names
        features_df = pd.DataFrame(input_data)

        # Pass the DataFrame directly to the pipeline
        prediction = model.predict(features_df)[0]

        return render_template(
            "index.html",
            prediction_text=f"Predicted CGPA: {round(float(prediction), 2)}"
        )

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)