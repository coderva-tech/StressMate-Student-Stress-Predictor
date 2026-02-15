import torch
import torch.nn as nn
import joblib
import numpy as np

# -----------------------------
# 1. Load scaler and label encoder
# -----------------------------
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

# -----------------------------
# 2. Neural network (same as training)
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
model.load_state_dict(torch.load('stress_model.pt'))
model.eval()

# -----------------------------
# 3. Input validation
# -----------------------------
def get_input(prompt, min_val, max_val):
    while True:
        try:
            val = float(input(prompt))
            if min_val <= val <= max_val:
                return val
            else:
                print(f"Please enter a value between {min_val} and {max_val}.")
        except ValueError:
            print("Invalid input. Enter a number.")

print("Enter student details (scale 1-5 or 1-7 where applicable):")
sleep_quality = get_input("Sleep Quality (1-5): ", 1, 5)
headache_freq = get_input("Headache Frequency per week (1-7): ", 1, 7)
academic_performance = get_input("Academic Performance (1-5): ", 1, 5)
study_load = get_input("Study Load (1-5): ", 1, 5)
extracurricular = get_input("Extracurricular Activities (1-5): ", 1, 5)

# -----------------------------
# 4. Prepare input
# -----------------------------
user_features = np.array([[sleep_quality, headache_freq, academic_performance, study_load, extracurricular]])
user_scaled = scaler.transform(user_features)
user_tensor = torch.tensor(user_scaled, dtype=torch.float32)

# -----------------------------
# 5. Predict
# -----------------------------
with torch.no_grad():
    outputs = model(user_tensor)
    probs = torch.softmax(outputs, dim=1).numpy()[0]
    predicted_idx = np.argmax(probs)
    predicted_category = le.inverse_transform([predicted_idx])[0]

# -----------------------------
# 6. Convert probabilities to numeric stress level (1-5)
# -----------------------------
# Assign representative numeric levels to classes
class_to_numeric = {'Low': 1.5, 'Medium': 3, 'High': 4.5}

# Weighted sum based on probabilities
predicted_level = probs[0]*class_to_numeric['Low'] + probs[1]*class_to_numeric['Medium'] + probs[2]*class_to_numeric['High']

# -----------------------------
# 7. Display results
# -----------------------------
print(f"\nPredicted Stress Category: {predicted_category}")
print(f"Predicted Stress Level (1-5 scale): {predicted_level:.2f}")
