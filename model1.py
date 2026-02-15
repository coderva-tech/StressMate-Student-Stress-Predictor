import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import joblib

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv('synthetic_student_stress.csv')

# -----------------------------
# 2. Create stress category
# -----------------------------
def stress_category(level):
    if level <= 2:
        return 'Low'
    elif level == 3:
        return 'Medium'
    else:
        return 'High'

df['stress_category'] = df['stress_level'].apply(stress_category)

# -----------------------------
# 3. Features and target
# -----------------------------
X = df[['sleep_quality', 'headache_freq', 'academic_performance', 'study_load', 'extracurricular']]
y = df['stress_category']

# Encode string labels to integers: Low=0, Medium=1, High=2
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# -----------------------------
# 4. Split dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# -----------------------------
# 5. Scale features
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 6. Convert to PyTorch tensors
# -----------------------------
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# -----------------------------
# 7. Define neural network
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

# -----------------------------
# 8. Loss and optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# -----------------------------
# 9. Training loop
# -----------------------------
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# -----------------------------
# 10. Save model and scaler
# -----------------------------
torch.save(model.state_dict(), 'stress_model.pt')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump((X_test, y_test), 'test_data.pkl')  # Save test data for evaluation
joblib.dump(le, 'label_encoder.pkl')           # Save LabelEncoder

print("Training complete. Model saved as stress_model.pt, scaler.pkl, and label_encoder.pkl")
