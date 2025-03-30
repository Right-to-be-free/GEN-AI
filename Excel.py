import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\rishi\Desktop\Random forest\archive (4).zip")
    df['mood_encoded'] = LabelEncoder().fit_transform(df['mood'])
    return df

df = load_data()

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

    features = ['steps', 'distance_km', 'calories_burned', 'active_minutes', 'sleep_hours', 'water_intake_liters']
    user_input = {}
    for feature in features:
        user_input[feature] = st.number_input(f"{feature.replace('_', ' ').title()}", min_value=0.0)

    # Prepare model
    X = df[features]
    y = df['mood_encoded']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    if st.button("Predict Mood"):
        input_df = pd.DataFrame([user_input])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        mood_label = LabelEncoder().fit(df['mood']).inverse_transform([prediction])[0]
        st.success(f"🎉 Predicted Mood: **{mood_label.upper()}**")

