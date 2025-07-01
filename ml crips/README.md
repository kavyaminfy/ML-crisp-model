
# 🏦 Loan Prediction Web App

This Flask application predicts whether a customer will opt for a **Personal Loan** based on input features uploaded via a CSV file. It uses a trained `VotingClassifier` (Logistic Regression + Random Forest + XGBoost), along with preprocessing steps like one-hot encoding, log transformation, scaling, and SMOTE for class balancing.

---

## 📦 Features

- Upload CSV files containing customer data
- Preprocesses data just like in training (log transform, encoding, scaling)
- Loads saved `.pkl` model, scaler, and column names
- Predicts loan uptake (`0` or `1`) with probability
- Displays results in an HTML table

---

## 🧠 Model Pipeline (Training Phase Summary)

- Used `VotingClassifier` combining:
  - Logistic Regression
  - Random Forest
  - XGBoost
- SMOTE applied on training data to handle class imbalance
- Features scaled using `StandardScaler`
- Model, scaler, and feature columns saved as `.pkl`

---



![image](https://github.com/user-attachments/assets/dc225b54-61b2-479f-b2a5-5b423eabd898)


![image](https://github.com/user-attachments/assets/a9927a7c-bfcc-48ac-a558-ce5e701d87a9)


