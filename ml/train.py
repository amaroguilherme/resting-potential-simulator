import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from ml.datasets import TrajectoryDataset
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

def evaluate(model, seq, pred_len, device):
    model.eval()
    preds = []
    seq = seq.to(device)
    with torch.no_grad():
        for _ in range(pred_len):
            y_pred = model(seq)
            preds.append(y_pred.item())
            new_step = y_pred.unsqueeze(0).unsqueeze(-1)
            seq = torch.cat([seq[:,1:,:], new_step], dim=1)
    return preds


num_traj = 50
traj_len = 11
X = np.linspace(-70, -40, traj_len).reshape(1, traj_len) + np.random.randn(num_traj, traj_len) * 2
Y = X.copy()
seq_len = 5
dataset = TrajectoryDataset(X, Y, seq_len=seq_len)
loader = DataLoader(dataset, batch_size=8, shuffle=True)


model1 = LSTMModel(input_dim=1, hidden_dim=32, num_layers=1).to(DEVICE)
optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(20):
    loss = train(model1, loader, optimizer1, criterion, DEVICE)
    print(f"[Exp 1] Epoch {epoch+1}, Loss: {loss:.4f}")

seq_input = torch.tensor(X[0,:seq_len], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
preds1 = evaluate(model1, seq_input, traj_len - seq_len, DEVICE)


model2 = LSTMModel(input_dim=1, hidden_dim=32, num_layers=2).to(DEVICE)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.01)

for epoch in range(20):
    loss = train(model2, loader, optimizer2, criterion, DEVICE)
    print(f"[Exp 2] Epoch {epoch+1}, Loss: {loss:.4f}")

seq_input2 = torch.tensor(X[0,:seq_len], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
preds2 = evaluate(model2, seq_input2, traj_len - seq_len, DEVICE)


plt.plot(np.arange(seq_len, traj_len), X[0, seq_len:], label="Ground Truth", marker='o')
plt.plot(np.arange(seq_len, traj_len), preds1, label="Prediction Exp1 (1 layer)", linestyle="--", marker='x')
plt.plot(np.arange(seq_len, traj_len), preds2, label="Prediction Exp2 (2 layers)", linestyle="--", marker='s')
plt.legend()
plt.xlabel("Timestep")
plt.ylabel("Value")
plt.title("LSTM results comparation")
plt.show()
