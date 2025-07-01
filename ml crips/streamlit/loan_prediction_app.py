import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

st.title("🔍 Outlier Detection using Logistic Regression")

# Upload the file
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Display data
    st.write("### Preview of Dataset")
    st.dataframe(df.head())

    # Drop 'ID' column if it exists
    df = df.drop(columns=['ID'], errors='ignore')

    # Detect outliers in 'Income'
    if 'Income' not in df.columns:
        st.error("⚠️ 'Income' column not found in the dataset.")
    else:
        Q1 = df['Income'].quantile(0.25)
        Q3 = df['Income'].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df['is_outlier'] = ((df['Income'] < lower) | (df['Income'] > upper)).astype(int)

        # Define features
        features = [col for col in df.columns if col != 'is_outlier']
        X = df[features]
        y = df['is_outlier']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train logistic regression model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.write("### Model Accuracy")
        st.write(f"{accuracy_score(y_test, y_pred):.2f}")
        st.write("### Classification Report")
        st.text(classification_report(y_test, y_pred))

        # Input for new prediction
        st.write("### Predict New Data Point")
        input_data = {}
        for col in features:
            val = st.number_input(f"Enter {col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
            input_data[col] = val

        input_df = pd.DataFrame([input_data])
        pred = model.predict(input_df)[0]
        st.write("### Prediction Result")
        st.success("Outlier" if pred == 1 else "Not an Outlier")

else:
    st.warning("👆 Please upload a CSV file to get started.")
