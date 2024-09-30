import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import re

# Load the datasets for iPhone and Samsung (replace with actual paths)
df_iphone = pd.read_csv('data/iphone_ebay.csv')
df_samsung = pd.read_csv('data/samsung_ebay.csv')

# Clean and preprocess price data (remove dollar signs, commas, handle ranges, and convert to float)
def clean_price(price):
    price = price.replace('$', '').replace(',', '')
    if 'to' in price:
        prices = price.split('to')
        prices = [float(p) for p in prices]
        return sum(prices) / len(prices)  # Average price if range
    else:
        return float(price)

df_iphone['price'] = df_iphone['price'].apply(clean_price)
df_samsung['price'] = df_samsung['price'].apply(clean_price)

# Add a 'platform' column and 'brand' column for distinction
df_iphone['platform'] = 'eBay'
df_samsung['platform'] = 'eBay'
df_iphone['brand'] = 'iPhone'
df_samsung['brand'] = 'Samsung'

# For demonstration, assign synthetic dates (replace with actual date if available)
df_iphone['date'] = pd.date_range(start='2023-01-01', periods=len(df_iphone), freq='D')
df_samsung['date'] = pd.date_range(start='2023-01-01', periods=len(df_samsung), freq='D')

# Combine both datasets
df_combined = pd.concat([df_iphone, df_samsung], ignore_index=True)

# Feature Engineering: extract model, storage, and color from the 'name' column
def extract_features(name):
    model = storage = color = 'Unknown'
    
    # Extract model (e.g., iPhone 13, Samsung S21)
    model_match = re.search(r'(iPhone|Samsung)\s(\w+\s*\w*)', name)
    if model_match:
        model = model_match.group(0)
    
    # Extract storage (e.g., 64GB, 128GB, etc.)
    storage_match = re.search(r'(\d+GB)', name)
    if storage_match:
        storage = storage_match.group(1)
    
    # Extract color
    color_match = re.search(r'-\s([\w\s]+)$', name)
    if color_match:
        color = color_match.group(1).strip()
    
    return pd.Series([model, storage, color])

df_combined[['model', 'storage', 'color']] = df_combined['name'].apply(extract_features)

# Label encode categorical features
categorical_cols = ['model', 'storage', 'color', 'condition', 'platform', 'brand']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_combined[col] = le.fit_transform(df_combined[col])
    label_encoders[col] = le

# Feature engineering: extract date features (day of week, month)
df_combined['day_of_week'] = df_combined['date'].dt.dayofweek
df_combined['month'] = df_combined['date'].dt.month

# Select features and target for the model
features = ['price', 'model', 'storage', 'color', 'condition', 'platform', 'brand', 'day_of_week', 'month']
target = 'price'

# Create sequences for time-series forecasting
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data.iloc[i:i+seq_length][features].values
        label = data.iloc[i+seq_length][target]
        sequences.append(seq)
        targets.append(label)
    return np.array(sequences), np.array(targets)

sequence_length = 7  # Use past 7 days to predict the next day
X, y = create_sequences(df_combined, sequence_length)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Scale the data using MinMaxScaler
num_samples, seq_length, num_features = X_train.shape
X_train_flat = X_train.reshape(-1, num_features)
X_test_flat = X_test.reshape(-1, num_features)

scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train_flat).reshape(num_samples, seq_length, num_features)
X_test_scaled = scaler_X.transform(X_test_flat).reshape(X_test.shape[0], seq_length, num_features)

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Define the LSTM model
class PricePredictorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(PricePredictorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze()

# Initialize the model
input_size = num_features
hidden_size = 64
model = PricePredictorLSTM(input_size, hidden_size)

# Convert the training data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 50
model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 5 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor).numpy()
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_actual = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

# Calculate metrics
mse = mean_squared_error(y_actual, y_pred)
mae = mean_absolute_error(y_actual, y_pred)
print(f'MSE: {mse:.2f}, MAE: {mae:.2f}')

# Improved Visualization
plt.figure(figsize=(14,7))
sns.set_style("whitegrid")

# Plot actual vs predicted prices
plt.plot(y_actual, label='Actual Prices', color='blue', linewidth=2, marker='o')
plt.plot(y_pred, label='Predicted Prices', color='green', linestyle='--', linewidth=2, marker='x')

# Add title and labels
plt.title('Actual vs Predicted Prices for iPhone & Samsung (eBay)', fontsize=16)
plt.xlabel('Time Steps (Days)', fontsize=12)
plt.ylabel('Price (in USD)', fontsize=12)
plt.legend(loc='upper right', fontsize=12)
plt.grid(True)
plt.show()
