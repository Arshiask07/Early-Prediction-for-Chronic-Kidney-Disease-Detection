from flask import Flask, request, render_template
import pandas as pd
import pickle
import logging
import os

app = Flask(__name__)
logging.basicConfig(level=logging.ERROR)

# Load model and encoders
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

model = pickle.load(open(os.path.join(MODEL_DIR, 'CKD.pkl'), 'rb'))
le_dict = pickle.load(open(os.path.join(MODEL_DIR, 'label_encoders.pkl'), 'rb'))
le_target = pickle.load(open(os.path.join(MODEL_DIR, 'target_encoder.pkl'), 'rb'))

# Expected input order
selected_features = [
    'specific_gravity',
    'albumin',
    'blood_glucose_random',
    'serum_creatinine',
    'haemoglobin',
    'packed_cell_volume',
    'red_blood_cell_count',
    'hypertension'
]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/Prediction', methods=['GET', 'POST'])
def prediction():
    return render_template('indexnew.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_dict = {}

        for feature in selected_features:
            input_dict[feature] = request.form[feature]

        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])

        # Convert numeric fields first
        for col in input_df.columns:
            if col not in le_dict:
                input_df[col] = input_df[col].astype(float)

        # Encode categorical fields
        for col in input_df.columns:
            if col in le_dict:
                input_df[col] = le_dict[col].transform(input_df[col])

        # Ensure correct feature order
        input_df = input_df[selected_features]

        # Prediction
        pred = model.predict(input_df)
        predicted_label = le_target.inverse_transform(pred)[0]

        return render_template(
            'result.html',
            prediction_text=predicted_label
        )

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return "Something went wrong while making the prediction. Please check your inputs and try again."


if __name__ == '__main__':
    app.run(debug=False)