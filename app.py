import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler
import joblib

# Define the same model architecture
class LeadScoringNN(nn.Module):
    def __init__(self, input_size):
        super(LeadScoringNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 8)
        self.fc6 = nn.Linear(8, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc6(x)  # Logits output (no sigmoid)
        return x

# Load the trained model

# Load the input size saved from training
input_size = joblib.load("input_size.pkl")
model = LeadScoringNN(input_size)
model.load_state_dict(torch.load("lead_scoring_model.pth"))
model.eval()

# Load MinMaxScaler to normalize input features
scaler = joblib.load("scaler.pkl")  # Save the scaler from training

# Initialize FastAPI
app = FastAPI()

# Define the request schema
class LeadData(BaseModel):
    features: list  # List of numerical feature values

@app.post("/predict")
def predict_conversion(data: LeadData):
    # Convert input data into a tensor
    features = np.array(data.features).reshape(1, -1)
    print(f"API received features: {len(data.features)}")

    features = scaler.transform(features)  # Normalize the input
    features_tensor = torch.tensor(features, dtype=torch.float32)
    
    # Make prediction
    with torch.no_grad():
        logits = model(features_tensor)
        probability = torch.sigmoid(logits).item()  # Convert logits to probability

    return {"conversion_probability": probability}
