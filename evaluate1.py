import torch
import torch.nn as nn
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------
# 1. Load scaler, test data, and label encoder
# -----------------------------
scaler = joblib.load('scaler.pkl')
X_test, y_test = joblib.load('test_data.pkl')
le = joblib.load('label_encoder.pkl')

# -----------------------------
# 2. Scale test features
# -----------------------------
X_test_scaled = scaler.transform(X_test)  # convert DataFrame to NumPy array if needed
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# -----------------------------
# 3. Define the same network architecture
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

# -----------------------------
# 4. Load trained model
# -----------------------------
model = StressNet()
model.load_state_dict(torch.load('stress_model.pt'))
model.eval()  # evaluation mode

# -----------------------------
# 5. Make predictions
# -----------------------------
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)

# -----------------------------
# 6. Evaluation metrics
# -----------------------------
accuracy = accuracy_score(y_test_tensor, predicted)
cm = confusion_matrix(y_test_tensor, predicted)
report = classification_report(y_test_tensor, predicted, target_names=le.classes_)

print(f"\nAccuracy: {accuracy*100:.2f}%")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)
