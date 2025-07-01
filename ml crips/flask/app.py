from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

#  Load model, scaler, and feature column names
model = joblib.load(r"C:\Users\Minfy.DESKTOP-81ME0ME\Desktop\ass 1 ml\flask\voting_classifier_model.pkl")
scaler = joblib.load(r"C:\Users\Minfy.DESKTOP-81ME0ME\Desktop\ass 1 ml\flask\scaler.pkl")
columns = joblib.load(r"C:\Users\Minfy.DESKTOP-81ME0ME\Desktop\ass 1 ml\flask\columns.pkl")

#  Log transformation list (if used during training)
log_features = ['Income', 'CCAvg', 'Mortgage']

@app.route('/')
def home():
    return render_template("upload.html")  # Upload page

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    try:
        # Read uploaded CSV
        file = request.files['file']
        df = pd.read_csv(file)

        # Optional: Apply log transformation if used during training
        for col in log_features:
            if col in df.columns:
                df[col] = np.log1p(df[col])

        # One-hot encode
        df_encoded = pd.get_dummies(df, drop_first=True)

        # Add missing columns with 0 and drop extras
        for col in columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[columns]

        # Scale using trained scaler
        X_scaled = scaler.transform(df_encoded)

        # Predict
        preds = model.predict(X_scaled)
        probs = model.predict_proba(X_scaled)[:, 1]

        # Add predictions to original DataFrame
        df['Prediction'] = preds
        df['Probability'] = probs.round(4)

        # Display results in HTML table
        return render_template("result.html", results=df.to_dict(orient='records'))

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
