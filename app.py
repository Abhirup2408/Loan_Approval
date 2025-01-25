import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import os

# Load dataset
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

data = pd.read_csv(r"C:\Users\ACER\Downloads\UniversalBank.csv")

# Preprocess data
def preprocess_data(data):
    X = data.drop(columns=["ID", "ZIP Code", "Personal Loan"])
    y = data["Personal Loan"]
    y = to_categorical(y)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, scaler

X, y, scaler = preprocess_data(data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build and save the model if not already saved
MODEL_PATH = "loan_model.h5"

@st.cache_resource
def build_and_train_model():
    if not os.path.exists(MODEL_PATH):
        model = Sequential()
        model.add(Dense(250, input_dim=X_train.shape[1], activation='relu', kernel_initializer='normal'))
        model.add(Dropout(0.3))
        model.add(Dense(500, activation='relu', kernel_initializer='normal'))
        model.add(Dropout(0.3))
        model.add(Dense(500, activation='relu', kernel_initializer='normal'))
        model.add(Dropout(0.3))
        model.add(Dense(250, activation='relu', kernel_initializer='normal'))
        model.add(Dropout(0.3))
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(X_train, y_train, validation_split=0.2, epochs=20, batch_size=32, verbose=0)

        # Save the model
        model.save(MODEL_PATH)
    else:
        model = load_model(MODEL_PATH)
    return model

model = build_and_train_model()

# Streamlit interface
st.title("Predict Personal Loan Acceptance")

st.sidebar.header("Input Parameters")
def user_input_features():
    age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
    experience = st.sidebar.number_input("Experience (Years)", min_value=0, max_value=50, value=5)
    income = st.sidebar.number_input("Income (in $000s)", min_value=0, max_value=300, value=50)
    family = st.sidebar.slider("Family Members", min_value=1, max_value=4, value=2)
    ccavg = st.sidebar.number_input("Average Credit Card Spending (in $000s)", min_value=0.0, max_value=10.0, value=2.0)
    education = st.sidebar.selectbox("Education Level", options=[1, 2, 3], index=1)
    mortgage = st.sidebar.number_input("Mortgage Value (in $000s)", min_value=0, max_value=700, value=0)
    securities_account = st.sidebar.selectbox("Securities Account", options=[0, 1], index=0)
    cd_account = st.sidebar.selectbox("CD Account", options=[0, 1], index=0)
    online = st.sidebar.selectbox("Online Banking", options=[0, 1], index=1)
    creditcard = st.sidebar.selectbox("Credit Card", options=[0, 1], index=1)

    data = {
        "Age": age,
        "Experience": experience,
        "Income": income,
        "Family": family,
        "CCAvg": ccavg,
        "Education": education,
        "Mortgage": mortgage,
        "Securities Account": securities_account,
        "CD Account": cd_account,
        "Online": online,
        "CreditCard": creditcard
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Scale user input
scaled_input = scaler.transform(input_df)

# Predict using the trained model
prediction = model.predict(scaled_input)
predicted_class = np.argmax(prediction, axis=1)[0]
predicted_probability = prediction[0][predicted_class]

# Display results
st.subheader("Prediction")
result = "Likely to accept a personal loan" if predicted_class == 1 else "Unlikely to accept a personal loan"
st.write(f"Prediction: **{result}**")
st.write(f"Confidence: **{predicted_probability:.2f}**")

st.subheader("Input Parameters")
st.write(input_df)
