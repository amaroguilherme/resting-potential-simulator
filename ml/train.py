import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, random_split
from ml.datasets import load_dataset, TrajectoryDataset
from ml.models import LSTMModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X_batch, Y_batch in loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        optimizer.zero_grad()
        Y_pred = model(X_batch)
        loss = criterion(Y_pred, Y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)

# ---------------
# Basic Training
# ---------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_traj = 50
    traj_len = 11
    X = np.linspace(-70, -40, traj_len).reshape(1, traj_len) + np.random.randn(num_traj, traj_len) * 2
    Y = X.copy()

    dataset = TrajectoryDataset(X, Y, seq_len=5)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = LSTMModel(input_dim=1, hidden_dim=32, num_layers=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(20):
        loss = train(model, loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    # -----------------
    # Basic Evaluation
    # -----------------
    model.eval()
    with torch.no_grad():
        seq = torch.tensor(X[0,:5], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        preds = []
        for _ in range(traj_len - 5):
            y_pred = model(seq)
            preds.append(y_pred.item())
            new_step = y_pred.unsqueeze(0).unsqueeze(-1)
            seq = torch.cat([seq[:,1:,:], new_step], dim=1)

    plt.plot(np.linspace(-70, -40, traj_len)[5:], label="Ground Truth")
    plt.plot(preds, label="Prediction", linestyle="--")
    plt.legend()
    plt.show()
