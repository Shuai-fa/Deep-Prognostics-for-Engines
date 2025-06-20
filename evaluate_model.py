# ==============================================================================
# 1. Import Libraries
# ==============================================================================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ==============================================================================
# 2. Define Model Architecture (Must be identical to training)
# ==============================================================================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_dim, n_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, 
                            batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

print("Model architecture defined.")

# ==============================================================================
# 3. Load the Trained Model
# ==============================================================================
# Define model parameters (must be identical to training)
input_size = 24
hidden_dim = 128
n_layers = 2
output_size = 1
sequence_length = 50

# Instantiate the model
device = torch.device('cpu') 
model = LSTMModel(input_size, hidden_dim, n_layers, output_size).to(device)

# Load the saved model weights
model_path = 'lstm_model.pth'
model.load_state_dict(torch.load(model_path, map_location=device))

# Switch to evaluation mode (very important!)
model.eval()

print(f"Model loaded from {model_path} and set to evaluation mode.")

# ==============================================================================
# 4. Load and Process Test Data
# ==============================================================================
print("Loading and processing test data...")
# Load test data and ground truth RUL labels
test_df = pd.read_csv('test_FD001.txt', sep='\s+', header=None)
truth_df = pd.read_csv('RUL_FD001.txt', sep='\s+', header=None)

test_df.dropna(axis=1, inplace=True)
truth_df.dropna(axis=1, inplace=True)

column_names = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor{i}' for i in range(1, 22)]
test_df.columns = column_names
truth_df.columns = ['RUL']

# --- CRITICAL: Use the same scaler fitted on the training data ---
# For simplicity, we re-load and fit the scaler on the training data here.
# In a production environment, the scaler object itself should be saved and loaded.
train_df = pd.read_csv('train_FD001.txt', sep='\s+', header=None)
train_df.dropna(axis=1, inplace=True)
train_df.columns = column_names
feature_columns = ['setting1', 'setting2', 'setting3'] + [f'sensor{i}' for i in range(1, 22)]
scaler = MinMaxScaler()
scaler.fit(train_df[feature_columns]) # Fit ONLY on training data

# Normalize the test data
test_df[feature_columns] = scaler.transform(test_df[feature_columns])

# --- Prepare input sequences for the test set (with padding) ---
test_sequences = []
for engine_id in test_df['engine_id'].unique():
    engine_data = test_df[test_df['engine_id'] == engine_id]
    engine_sequence = engine_data[feature_columns].values
    
    if len(engine_sequence) < sequence_length:
        padding_rows = sequence_length - len(engine_sequence)
        padding_zeros = np.zeros((padding_rows, engine_sequence.shape[1]))
        final_sequence = np.concatenate((padding_zeros, engine_sequence), axis=0)
    else:
        final_sequence = engine_sequence[-sequence_length:]
        
    test_sequences.append(final_sequence)

test_sequences = np.array(test_sequences)
X_test = torch.from_numpy(test_sequences).float()

print("Test data processing complete.")


# ==============================================================================
# 5. Make Predictions
# ==============================================================================
print("Making predictions on the test set...")
with torch.no_grad(): # No need to calculate gradients during evaluation
    y_pred = model(X_test.to(device))

predicted_ruls = y_pred.cpu().numpy().flatten()
true_ruls = truth_df['RUL'].values

# ==============================================================================
# 6. Evaluate and Visualize Results
# ==============================================================================
# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(np.mean((predicted_ruls - true_ruls)**2))
print(f"\nRoot Mean Squared Error (RMSE) on Test Set: {rmse:.2f}")

# Visualize the comparison
plt.figure(figsize=(14, 7))
plt.plot(true_ruls, label='True RUL', color='blue', marker='o', linestyle='None', markersize=6)
plt.plot(predicted_ruls, label='Predicted RUL', color='red', marker='x', linestyle='None', markersize=6)
plt.title('True vs. Predicted RUL on Test Set', fontsize=16)
plt.xlabel('Engine ID', fontsize=12)
plt.ylabel('Remaining Useful Life (RUL)', fontsize=12)
plt.legend()
plt.grid()
plt.show()