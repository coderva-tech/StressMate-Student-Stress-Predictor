import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np
import random

# -----------------------------
# Load scaler and label encoder
# -----------------------------
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

# -----------------------------
# Neural network
# -----------------------------
class StressNet(nn.Module):
    def __init__(self):
        super(StressNet, self).__init__()
        self.fc1 = nn.Linear(5, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 3)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

model = StressNet()
model.load_state_dict(torch.load('stress_model.pt', map_location=torch.device('cpu')))
model.eval()

# -----------------------------
# Numeric mapping
# -----------------------------
class_to_numeric = {'Low': 1.5, 'Medium': 3, 'High': 4.5}

# Commentary for stress categories
stress_comments = {
    'Low': [
        "You're doing great! Keep maintaining your routine and healthy habits.",
        "Stress is low. Keep up the balance and stay consistent!"
    ],
    'Medium': [
        "Moderate stress detected. Consider short breaks or relaxation exercises.",
        "Try to manage workload and get enough sleep to stay balanced."
    ],
    'High': [
        "High stress detected! Take a break, relax, and practice self-care.",
        "Consider talking to a counselor or practicing mindfulness exercises."
    ]
}

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🧠 Student Stress Predictor")
st.markdown("Enter your daily or weekly details to predict your stress level.")

# Input sliders
sleep_quality = st.slider("Sleep Quality (1-5)", 1, 5, 3)
headache_freq = st.slider("Headache Frequency per week (1-7)", 1, 7, 3)
academic_performance = st.slider("Academic Performance (1-5)", 1, 5, 3)
study_load = st.slider("Study Load (1-5)", 1, 5, 3)
extracurricular = st.slider("Extracurricular Activities (1-5)", 1, 5, 3)

if st.button("Predict Stress"):
    # -----------------------------
    # Prepare input
    # -----------------------------
    user_features = np.array([[sleep_quality, headache_freq, academic_performance, study_load, extracurricular]])
    user_scaled = scaler.transform(user_features)
    user_tensor = torch.tensor(user_scaled, dtype=torch.float32)
    
    # -----------------------------
    # Predict
    # -----------------------------
    with torch.no_grad():
        outputs = model(user_tensor)
        probs = torch.softmax(outputs, dim=1).numpy()[0]
        predicted_idx = np.argmax(probs)
        predicted_category = le.inverse_transform([predicted_idx])[0]
        predicted_level = sum(probs[i]*class_to_numeric[c] for i, c in enumerate(['Low','Medium','High']))
    
    # -----------------------------
    # Display results
    # -----------------------------
    st.success(f"Predicted Stress Category: **{predicted_category}**")
    st.info(f"Predicted Stress Level (1-5 scale): **{predicted_level:.2f}**")
    
    # -----------------------------
    # Display dynamic commentary
    # -----------------------------
    comment = random.choice(stress_comments[predicted_category])
    st.write("💬", comment)
    
    # -----------------------------
    # Factor-based tips
    # -----------------------------
    if sleep_quality <= 2:
        st.warning("⚠️ Your sleep quality is low. Aim for 7-8 hours of good sleep.")
    if study_load >= 4:
        st.warning("⚠️ High study load detected. Break tasks into smaller chunks and take breaks.")
    if headache_freq >= 5:
        st.warning("⚠️ Frequent headaches! Stay hydrated and rest your eyes.")
    if extracurricular <= 2:
        st.info("💡 Engaging in extracurricular activities can help reduce stress.")
    
    # -----------------------------
    # Probability-based tip
    # -----------------------------
    if probs[2] > 0.6:  # High stress probability
        st.error("🚨 High stress probability detected. Take immediate measures to relax!")
    elif probs[1] > 0.5:  # Medium stress probability
        st.warning("⚠️ Moderate stress probability. Watch your workload and rest adequately.")
    else:
        st.success("✅ Stress probability is low. Keep maintaining healthy routines!")
    
    # -----------------------------
    # Optional resource links
    # -----------------------------
    if predicted_category == 'High':
        st.markdown("[🧘 Try a 5-minute guided meditation](https://www.headspace.com/meditation)")
        st.markdown("[📖 Read tips to manage stress effectively](https://www.mentalhealth.org.uk/a-to-z/s/stress)")
