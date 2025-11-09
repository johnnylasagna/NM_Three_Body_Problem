# Imports
# - pandas: for data manipulation and reading CSV files
# - numpy: for numerical operations
# - torch: for building and training neural networks
# - sklearn: for data preprocessing (e.g., scaling)
# - matplotlib: for plotting results
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

try:
    # neuraloperator: library for Fourier Neural Operators (FNOs)
    from neuralop.models import TFNO
    from neuralop.training import Trainer
    from neuralop.losses import LpLoss
except ImportError:
    print("neuraloperator library not found.")
    print("Please install it using: pip install neuraloperator")
    exit()

# Dataset class
# Custom PyTorch Dataset to handle time-series data for training FNOs
class DictTensorDataset(Dataset):
    def __init__(self, data, n_steps):
        # data: scaled time-series data
        # n_steps: number of timesteps used for input/output sequences
        self.data = data
        self.n_steps = n_steps

    def __len__(self):
        # Total sequences available for training
        return len(self.data) - 2 * self.n_steps + 1

    def __getitem__(self, idx):
        # Extract input (x) and output (y) sequences
        x = self.data[idx:(idx + self.n_steps)]
        y = self.data[(idx + self.n_steps):(idx + 2 * self.n_steps)]
        # Convert to PyTorch tensors and permute dimensions for FNO
        x_tensor = torch.tensor(x, dtype=torch.float32).permute(1, 0)
        y_tensor = torch.tensor(y, dtype=torch.float32).permute(1, 0)
        return {'x': x_tensor, 'y': y_tensor}

# Select device for computation (GPU if available, else CPU)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# Parameters for the FNO model and training
N_STEPS = 10  # Number of timesteps in input/output sequences
N_MODES = (16,)  # Fourier modes for FNO layers
HIDDEN_CHANNELS = 64  # Number of hidden channels in FNO
N_LAYERS = 4  # Number of FNO layers
BATCH_SIZE = 256  # Batch size for training
N_EPOCHS = 20  # Number of training epochs
LEARNING_RATE = 5e-4  # Learning rate for optimizer
WEIGHT_DECAY = 1e-5  # Weight decay for regularization

# Load dataset and preprocess it
# Returns test dataset, scaled data, scaler, and number of features
def load_data():
    print("Loading dataset...")
    df = pd.read_csv('Three_Body_Problem/data/TBP_dataset.csv')
    print(f"Original data shape: {df.shape}")
    N_TIMESTEPS, N_FEATURES = df.shape
    time_series_data = df.values
    # Split data into train, validation, and test sets (70/15/15 split)
    train_size = int(len(time_series_data) * 0.7)
    val_size = int(len(time_series_data) * 0.15)
    train_data = time_series_data[:train_size]
    val_data = time_series_data[train_size:train_size + val_size]
    test_data = time_series_data[train_size + val_size:]
    # Scale data to zero mean and unit variance
    scaler = StandardScaler()
    scaler.fit(train_data)
    test_data_scaled = scaler.transform(test_data)
    # Wrap test data in custom Dataset class
    test_dataset = DictTensorDataset(test_data_scaled, N_STEPS)
    return test_dataset, test_data_scaled, scaler, N_FEATURES

# Load pre-trained FNO model
# Returns the model in evaluation mode
def load_model(in_channels, out_channels):
    print("Loading the FNO model...")
    model = TFNO(
        n_modes=N_MODES,
        hidden_channels=HIDDEN_CHANNELS,
        in_channels=in_channels,
        out_channels=out_channels,
        n_layers=N_LAYERS,
        use_mlp=True,
        mlp={'expansion': 2.0, 'dropout': 0.1},
        lifting_channels=256,
        projection_channels=256
    ).to(DEVICE)
    model_save_path = 'Three_Body_Problem/FNO/saved_model_2.pth'
    try:
        # Load model weights from file
        model.load_state_dict(torch.load(model_save_path, map_location=DEVICE, weights_only=False))
        print(f"Model loaded from {model_save_path}")
    except FileNotFoundError:
        print(f"Model file not found at {model_save_path}. Please train the model first by running `train()`.")
        exit()
    model.eval()
    return model

# Training function for the FNO model
def train():
    print("Loading dataset...")
    df = pd.read_csv('Three_Body_Problem/data/TBP_dataset.csv')
    print(f"Original data shape: {df.shape}")
    N_TIMESTEPS, N_FEATURES = df.shape
    IN_CHANNELS = N_FEATURES
    OUT_CHANNELS = N_FEATURES
    time_series_data = df.values
    # Split data into train, validation, and test sets
    train_size = int(len(time_series_data) * 0.7)
    val_size = int(len(time_series_data) * 0.15)
    test_size = len(time_series_data) - train_size - val_size
    train_data = time_series_data[:train_size]
    val_data = time_series_data[train_size:train_size + val_size]
    test_data = time_series_data[train_size + val_size:]
    print(f"Train data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    # Scale data
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    val_data_scaled = scaler.transform(val_data)
    test_data_scaled = scaler.transform(test_data)
    # Wrap data in custom Dataset class
    train_dataset = DictTensorDataset(train_data_scaled, N_STEPS)
    val_dataset = DictTensorDataset(val_data_scaled, N_STEPS)
    test_dataset = DictTensorDataset(test_data_scaled, N_STEPS)
    # Create DataLoaders for batching
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print("Defining the FNO model...")
    # Initialize FNO model
    model = TFNO(
        n_modes=N_MODES,
        hidden_channels=HIDDEN_CHANNELS,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        n_layers=N_LAYERS,
        use_mlp=True,
        mlp={'expansion': 2.0, 'dropout': 0.1},
        lifting_channels=256,
        projection_channels=256
    ).to(DEVICE)

    print(model)

    # Optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    # Loss function (L2 loss)
    l2loss = LpLoss(d=2, p=2)

    print("Setting up the trainer...")
    # Trainer handles training and validation
    trainer = Trainer(
        model=model,
        n_epochs=N_EPOCHS,
        device=DEVICE,
        wandb_log=False,
        use_distributed=False,
        verbose=True
    )

    print("Starting training...")
    # Train the model
    trainer.train(
        train_loader=train_loader,
        test_loaders={'validation': val_loader},
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=None,
        training_loss=l2loss,
        eval_losses={'validation_l2': l2loss}
    )

    # Save trained model weights
    model_save_path = 'Three_Body_Problem/library_implementation/saved_model_2.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Perform autoregressive rollout on test data
    print("Performing autoregressive rollout on a test sample...")
    model.eval()
    first_sample = test_dataset[0]['x'].unsqueeze(0).to(DEVICE)
    initial_sequence = first_sample
    ground_truth_rollout = test_data_scaled
    rollout_steps = 200
    predictions = []
    current_sequence = initial_sequence.clone()

    with torch.no_grad():
        for i in range(rollout_steps):
            # Predict next step and append to predictions
            pred = model(current_sequence)
            next_step_pred = pred[:, :, 0:1]
            predictions.append(next_step_pred.cpu().numpy().squeeze())
            # Update current sequence with new prediction
            current_sequence = torch.cat([current_sequence[:, :, 1:], next_step_pred], dim=2)

    # Rescale predictions and ground truth for plotting
    predictions = np.array(predictions)
    predictions_rescaled = scaler.inverse_transform(predictions)
    ground_truth_rescaled = scaler.inverse_transform(ground_truth_rollout[:rollout_steps])
    print("Plotting results...")
    # Plot predictions vs ground truth for selected features
    features_to_plot = [0, 1, 2]
    fig, axes = plt.subplots(len(features_to_plot), 1, figsize=(15, 5 * len(features_to_plot)), sharex=True)
    fig.suptitle('Autoregressive Rollout vs. Ground Truth', fontsize=16)

    for i, feature_idx in enumerate(features_to_plot):
        ax = axes[i]
        ax.plot(ground_truth_rescaled[:, feature_idx], label='Ground Truth', color='blue')
        ax.plot(predictions_rescaled[:, feature_idx], label='FNO Prediction', color='red', linestyle='--')
        ax.set_ylabel(f'Feature {feature_idx}')
        ax.legend()
        ax.grid(True)

    axes[-1].set_xlabel('Time Steps into the Future')
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()

# Entry point for training
# Ensures num_workers=4 works correctly when running the script
if __name__ == '__main__':
    train()
