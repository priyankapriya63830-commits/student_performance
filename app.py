import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns


# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Student Performance App",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Prediction System")


# ----------------------------
# LOAD DATA
# ----------------------------
DATA_PATH = "data/StudentPerformanceFactors.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates()
    df = df.dropna()
    return df


data = load_data()

st.subheader("📊 Dataset Preview")
st.dataframe(data.head())


# ----------------------------
# FEATURE ENGINEERING
# ----------------------------
data["Study_Efficiency"] = (
    data["Hours_Studied"] * data["Attendance"] / 100
)

features = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Physical_Activity",
    "Study_Efficiency"
]

target = "Exam_Score"

X = data[features]
y = data[target]


# ----------------------------
# TRAIN MODEL
# ----------------------------
def train_model():

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    return model, scaler, mae, rmse, r2


# ----------------------------
# TRAIN BUTTON
# ----------------------------
if st.button("🚀 Train Model"):

    model, scaler, mae, rmse, r2 = train_model()

    st.success("Model trained successfully!")

    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("RMSE", f"{rmse:.2f}")
    col3.metric("R² Score", f"{r2:.2f}")

# ----------------------------
# CORRELATION HEATMAP
# ----------------------------
st.subheader("🔥 Correlation Heatmap")

fig, ax = plt.subplots(figsize=(10, 6))

sns.heatmap(
    data[features + [target]].corr(),
    annot=True,
    cmap="coolwarm",
    ax=ax
)

st.pyplot(fig)


# ----------------------------
# PREDICTION SECTION
# ----------------------------
st.subheader("🎯 Predict Exam Score")

hours = st.number_input("Hours Studied", 0.0, 24.0, 5.0)
attendance = st.number_input("Attendance (%)", 0.0, 100.0, 80.0)
sleep = st.number_input("Sleep Hours", 0.0, 12.0, 7.0)
previous = st.number_input("Previous Scores", 0.0, 100.0, 70.0)
tutoring = st.number_input("Tutoring Sessions", 0, 10, 2)
physical = st.number_input("Physical Activity", 0, 10, 3)

study_eff = hours * attendance / 100

if st.button("📈 Predict Score"):

    try:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")

        input_data = pd.DataFrame([[
            hours,
            attendance,
            sleep,
            previous, 
            tutoring,
            physical,
            study_eff
        ]], columns=features)

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]

        st.success(f"🎯 Predicted Exam Score: {prediction:.2f}")

    except:
        st.error("Please train the model first before predicting.")


# ----------------------------
# DATA INSIGHTS
# ----------------------------
st.subheader("📌 Quick Insights")

col1, col2, col3 = st.columns(3)

col1.metric("Total Students", len(data))
col2.metric("Avg Exam Score", f"{data['Exam_Score'].mean():.2f}")
col3.metric("Max Score", f"{data['Exam_Score'].max():.2f}")