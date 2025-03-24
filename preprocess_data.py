import pandas as pd

# Load the dataset
df = pd.read_csv("synthetic_lead_scoring_data.csv")  # Use the correct file path
df.head()

# Check for missing values (columns)
missing_values = df.isnull().sum()
print(missing_values)

# Display only rows that contain missing values
df[df.isnull().any(axis=1)]

#Visualize missing data
import missingno as msno
import matplotlib.pyplot as plt
# Visualize missing values
msno.matrix(df)
plt.show()

# Calculate percentage of missing values
missing_percentage = (df.isnull().sum() / len(df)) * 100
print(missing_percentage)

# Count missing values per row
df["missing_count"] = df.isnull().sum(axis=1)

# Show rows with missing values
df[df["missing_count"] > 0].head(10)

# Compare missing values across industries
print(df.groupby("industry").apply(lambda x: x.isnull().sum()))

# Drop all rows with any missing values
df_cleaned = df.dropna()

# Check the number of rows after dropping
print(f"Total rows after dropping missing values: {len(df_cleaned)}")

#Final Check: Confirm No More Missing Data
print(df_cleaned.isnull().sum())  # Should print all zeros
print(df_cleaned.columns)
df["missing_count"] = df.isnull().sum(axis=1)
df_cleaned.drop(columns=["missing_count"], inplace=True, errors="ignore")

#################################################################
# Apply one-hot encoding while dropping the first category to prevent redundancy
df_cleaned = pd.get_dummies(df_cleaned, columns=["industry", "company_size"])

# Check the new column names
print(df_cleaned.columns)

#Normalize numerical values and split dataset for training a NN
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Define features and target
X = df_cleaned.drop(columns=["conversion"]).values  # Convert to NumPy array
y = df_cleaned["conversion"].values

# Apply Min-Max Scaling
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # Reshape for PyTorch
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)  # Reshape for PyTorch

#Define the Neural Network

class LeadScoringNN(nn.Module):
    def __init__(self):
        super(LeadScoringNN, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 8)   # New layer
        self.fc6 = nn.Linear(8, 1)  # New final output layer
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
        x = self.fc6(x)
        return x


# Re-initialize model
model = LeadScoringNN()



#Train the Model
# Define loss function and optimizer
# Compute class weights
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Compute class weights manually
class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor([1.0, 2.5], dtype=torch.float32)  # Slightly reduce class 1 weight

# Apply new weights to loss function
pos_weight = torch.tensor([1.5], dtype=torch.float32)  # Fix class weight handling
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  



optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.005)  # Increase L1 regularization


scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)




# Reduce LR by half every 50 epochs

epochs = 250  # Train even longer
#batch_size = 16

for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    loss.backward()
    optimizer.step()
    scheduler.step(loss.item())  # Adjust learning rate

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']}")

#Evaluate the Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Predict on test data
# Adjust classification threshold
threshold = 0.625  # Increase from 0.5 to 0.6
with torch.no_grad():  # No need to track gradients
    y_pred_probs = torch.sigmoid(model(X_test_tensor))
    y_pred = (y_pred_probs > threshold).float()   # Convert probabilities to 0 or 1

# Convert tensors to NumPy for evaluation
y_pred_np = y_pred.numpy()
y_test_np = y_test_tensor.numpy()

# Compute metrics
accuracy = accuracy_score(y_test_np, y_pred_np)
precision = precision_score(y_test_np, y_pred_np)
recall = recall_score(y_test_np, y_pred_np)
f1 = f1_score(y_test_np, y_pred_np)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

#Save Model
torch.save(model.state_dict(), "lead_scoring_model.pth")
print("Model saved successfully!")

print(f"Training features: {df_cleaned.drop(columns=['conversion']).columns.tolist()}")
print(f"Number of features in training: {X_train.shape[1]}")


import joblib

# Save the input size
joblib.dump(X_train.shape[1], "input_size.pkl")

# Save the MinMaxScaler for preprocessing in API
joblib.dump(scaler, "scaler.pkl")
