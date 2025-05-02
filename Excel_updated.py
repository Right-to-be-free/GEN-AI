# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\\Users\\rishi\\Desktop\\Random forest\\archive (4)\\steps_tracker_dataset.csv")

    # Add synthetic extended features
    np.random.seed(42)
    df['mood_last_day'] = np.random.choice(['happy', 'sad', 'tired', 'stressed', 'energetic'], len(df))
    df['weekend'] = np.random.choice([0, 1], len(df))
    df['gender'] = np.random.choice(['male', 'female', 'other'], len(df))
    df['age'] = np.random.randint(18, 60, size=len(df))
    df['heart_rate'] = np.random.randint(60, 100, size=len(df))

    le_mood = LabelEncoder()
    le_last_day = LabelEncoder()
    le_gender = LabelEncoder()

    df['mood_encoded'] = le_mood.fit_transform(df['mood'])
    df['mood_last_day'] = le_last_day.fit_transform(df['mood_last_day'])
    df['gender'] = le_gender.fit_transform(df['gender'])

    return df, le_mood, le_last_day, le_gender

df, le, le_last_day, le_gender = load_data()

# Define features
features = [
    'steps', 'distance_km', 'calories_burned', 'active_minutes',
    'sleep_hours', 'water_intake_liters', 'mood_last_day', 'weekend',
    'age', 'gender', 'heart_rate'
]

# Sidebar
st.sidebar.title("Mood Prediction Dashboard")
section = st.sidebar.radio("Go to", ["📊 Data Overview", "📈 Model & Prediction"])

# Main Title
st.title("🧠 Mood Predictor from Daily Activities")

# --- Section 1: Data Overview ---
if section == "📊 Data Overview":
    st.subheader("Raw Data Sample")
    st.write(df.head())

    st.subheader("Mood Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x="mood", data=df, order=df["mood"].value_counts().index, ax=ax1)
    st.pyplot(fig1)

    st.subheader("Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.select_dtypes(include="number").corr(), annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

# --- Section 2: Model and Prediction ---
elif section == "📈 Model & Prediction":
    st.subheader("Enter Daily Stats to Predict Mood")

    # Inputs
    user_input = {
        'steps': st.number_input("Steps", min_value=0.0),
        'distance_km': st.number_input("Distance (km)", min_value=0.0),
        'calories_burned': st.number_input("Calories Burned", min_value=0.0),
        'active_minutes': st.number_input("Active Minutes", min_value=0.0),
        'sleep_hours': st.number_input("Sleep Hours", min_value=0.0),
        'water_intake_liters': st.number_input("Water Intake (liters)", min_value=0.0),
        'heart_rate': st.number_input("Heart Rate", min_value=40.0, max_value=200.0, value=80.0),
        'age': st.slider("Age", 13, 80, 25),
        'mood_last_day': le_last_day.transform([st.selectbox("Mood Yesterday", le_last_day.classes_)])[0],
        'gender': le_gender.transform([st.selectbox("Gender", le_gender.classes_)])[0],
        'weekend': int(st.checkbox("Is it a Weekend?"))
    }

    # Model selector
    model_type = st.selectbox("Choose Model", ["Random Forest", "Decision Tree", "Logistic Regression"])

    # Prepare data
    X = df[features]
    y = df['mood_encoded']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Model setup
    if model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "Decision Tree":
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
    else:
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)

    model.fit(X_train, y_train)
    score = accuracy_score(y_test, model.predict(X_test))
    st.metric(label="📊 Model Accuracy (test)", value=f"{score:.2%}")

    # Prediction
    if st.button("Predict Mood"):
        input_df = pd.DataFrame([user_input])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        mood_label = le.inverse_transform([prediction])[0]
        st.success(f"🎉 Predicted Mood: **{mood_label.upper()}**")

# --- Bulk Upload ---
st.subheader("📥 Bulk Upload: Predict Moods from CSV")
uploaded_file = st.file_uploader("Upload CSV file with daily stats", type=["csv"])
if uploaded_file:
    try:
        user_df = pd.read_csv(uploaded_file)
        st.write("📄 Uploaded Data:", user_df.head())

        # Encode categorical fields if needed
        if 'mood_last_day' in user_df.columns:
            user_df['mood_last_day'] = le_last_day.transform(user_df['mood_last_day'])
        if 'gender' in user_df.columns:
            user_df['gender'] = le_gender.transform(user_df['gender'])
        if 'weekend' in user_df.columns:
            user_df['weekend'] = user_df['weekend'].astype(int)

        if all(col in user_df.columns for col in features):
            user_scaled = scaler.transform(user_df[features])
            batch_preds = model.predict(user_scaled)
            user_df['Predicted Mood'] = le.inverse_transform(batch_preds)
            st.success("✅ Mood predictions added!")
            st.write(user_df)

            csv_download = user_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Predictions as CSV", data=csv_download, file_name="predicted_moods.csv", mime='text/csv')
        else:
            st.error(f"❌ CSV must contain columns: {features}")
    except Exception as e:
        st.error(f"Error reading file: {e}")
