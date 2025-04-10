from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)


# Route for home page
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/home')
def author_details():
    return render_template('home.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    input_features = [float(x) for x in request.form.values()]
    feature_names = ['rcount', 'gender', 'dialysisrenalendstage', 'asthma', 'irondef',
                     'pneum', 'substancedependence', 'psychologicaldisordermajor',
                     'depress', 'psychother', 'fibrosisandother', 'malnutrition', 'hemo',
                     'hematocrit', 'neutrophils', 'sodium', 'glucose', 'bloodureanitro',
                     'creatinine', 'bmi', 'pulse', 'respiration',
                     'secondarydiagnosisnonicd9', 'facid']

    # Convert features to DataFrame for model input
    features = pd.DataFrame([input_features], columns=feature_names)

    # Predict using the model
    prediction = model.predict(features)[0]

    return render_template('index.html', prediction_text=f'Predicted Length of Stay: {prediction:.0f} days')


if __name__ == '__main__':
    app.run(debug=True)
