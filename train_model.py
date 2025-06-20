# ==============================================================================
# 1. Import Libraries
# ==============================================================================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import time

print("Libraries imported successfully.")

# ==============================================================================
# 2. Define Hyperparameters and Global Settings
# ==============================================================================
# Data processing parameters
sequence_length = 50 
batch_size = 256     

# Model architecture parameters
input_size = 24      # Number of features (3 settings + 21 sensors)
hidden_dim = 128     
n_layers = 2         
output_size = 1      

# Training parameters
num_epochs = 100     # Increased for better performance
learning_rate = 0.001 

# ==============================================================================
# 3. Define the PyTorch LSTM Model
# ==============================================================================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_dim, n_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, 
                            batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

print("LSTM model definition complete.")


# ==============================================================================
# 4. Load and Preprocess Data (Robust Version)
# ==============================================================================
print("Loading and preprocessing data...")
start_time = time.time()

# --- Load data using a robust method ---
data_path = './'
train_file = data_path + 'train_FD001.txt'
column_names = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor{i}' for i in range(1, 22)]

# Use regex '\s+' as a separator to handle one or more spaces
train_df = pd.read_csv(train_file, sep='\s+', header=None)
train_df.dropna(axis=1, inplace=True) # Drop trailing empty columns
train_df.columns = column_names # Assign column names after cleaning

# Calculate RUL
max_cycles = train_df.groupby('engine_id')['cycle'].max().reset_index()
max_cycles.columns = ['engine_id', 'max_cycle']
train_df = pd.merge(train_df, max_cycles, on='engine_id', how='left')
train_df['RUL'] = train_df['max_cycle'] - train_df['cycle']
train_df.drop(columns=['max_cycle'], inplace=True)

# Clip RUL at 125 for better model training
train_df['RUL'] = train_df['RUL'].clip(upper=125)

# Normalize features
feature_columns = ['setting1', 'setting2', 'setting3'] + [f'sensor{i}' for i in range(1, 22)]
scaler = MinMaxScaler()
train_df[feature_columns] = scaler.fit_transform(train_df[feature_columns])

# --- Create sequences using a sliding window ---
sequences = []
targets = []
for engine_id in train_df['engine_id'].unique():
    engine_data = train_df[train_df['engine_id'] == engine_id]
    engine_features = engine_data[feature_columns].values
    engine_rul = engine_data['RUL'].values
    for i in range(len(engine_data) - sequence_length):
        sequences.append(engine_features[i:i + sequence_length])
        targets.append(engine_rul[i + sequence_length - 1])

X_train = np.array(sequences)
y_train = np.array(targets)

# Convert to PyTorch Tensors
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float()

# Create TensorDataset and DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

print(f"Data processing complete. Time elapsed: {time.time() - start_time:.2f}s")
print(f"Training data shape: {X_train_tensor.shape}")
print(f"Target data shape: {y_train_tensor.shape}")


# ==============================================================================
# 5. Initialize Model, Loss Function, and Optimizer
# ==============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training will use device: {device}")

model = LSTMModel(input_size, hidden_dim, n_layers, output_size).to(device)
criterion = nn.MSELoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("Model, Loss Function, and Optimizer initialized.")

# ==============================================================================
# 6. Train the Model
# ==============================================================================
print("\nStarting model training...")
start_time = time.time()

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 10 == 0: # Print loss every 10 epochs
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f"Model training complete. Total time: {time.time() - start_time:.2f}s")

# ==============================================================================
# 7. Save the Trained Model
# ==============================================================================
model_path = 'lstm_model.pth'
torch.save(model.state_dict(), model_path)
print(f"\nModel saved to: {model_path}")