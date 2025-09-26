import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from ml.datasets import TrajectoryDataset
from ml.models import LSTMModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for X_batch, Y_batch in loader:
        if X_batch.ndim == 2:
            X_batch = X_batch.unsqueeze(-1)
        if Y_batch.ndim == 1:
            Y_batch = Y_batch.unsqueeze(-1)

        X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)

        optimizer.zero_grad()
        Y_pred = model(X_batch)

        Y_target = Y_batch[:, -1] if Y_batch.ndim == 2 else Y_batch[:, -1, :]
        loss = criterion(Y_pred, Y_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, Y_batch in loader:
            if X_batch.ndim == 2:
                X_batch = X_batch.unsqueeze(-1)
            if Y_batch.ndim == 1:
                Y_batch = Y_batch.unsqueeze(-1)

            X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
            Y_pred = model(X_batch)

            Y_target = Y_batch[:, -1] if Y_batch.ndim == 2 else Y_batch[:, -1, :]
            loss = criterion(Y_pred, Y_target)
            total_loss += loss.item() * X_batch.size(0)

            all_preds.append(Y_pred.cpu().numpy())
            all_targets.append(Y_target.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    rmse = math.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)
    evs = explained_variance_score(targets, preds)

    return total_loss / len(loader.dataset), rmse, mae, r2, evs, preds, targets



def run_experiment(config, X, Y):
    dataset = TrajectoryDataset(X, Y, seq_len=5)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8)

    model = LSTMModel(
        input_dim=1,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"]
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val = float("inf")
    patience, wait = 5, 0
    for _ in range(30):
        _ = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_rmse, val_mae, val_r2, val_evs, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    val_loss, val_rmse, val_mae, val_r2, val_evs, preds, targets = evaluate(model, val_loader, criterion)

    return {
        "hidden_dim": config["hidden_dim"],
        "num_layers": config["num_layers"],
        "lr": config["lr"],
        "dropout": config["dropout"],
        "weight_decay": config["weight_decay"],
        "val_loss": val_loss,
        "val_rmse": val_rmse,
        "val_mae": val_mae,
        "r2": val_r2,
        "explained_variance": val_evs,
        "preds": preds.tolist(),
        "targets": targets.tolist()
    }

if __name__ == "__main__":
    # synthetic dataset
    num_traj = 50
    traj_len = 11
    X = np.linspace(-70, -40, traj_len).reshape(1, traj_len) + np.random.randn(num_traj, traj_len) * 2
    Y = X.copy()

    configs = [
        {"hidden_dim": 32, "num_layers": 1, "lr": 1e-3, "dropout": 0.0, "weight_decay": 0},
        {"hidden_dim": 32, "num_layers": 2, "lr": 1e-3, "dropout": 0.0, "weight_decay": 0},
        {"hidden_dim": 64, "num_layers": 1, "lr": 1e-3, "dropout": 0.0, "weight_decay": 0},
        {"hidden_dim": 64, "num_layers": 2, "lr": 1e-3, "dropout": 0.0, "weight_decay": 0},
        {"hidden_dim": 32, "num_layers": 1, "lr": 5e-4, "dropout": 0.1, "weight_decay": 1e-5},
        {"hidden_dim": 32, "num_layers": 2, "lr": 5e-4, "dropout": 0.1, "weight_decay": 1e-5},
        {"hidden_dim": 64, "num_layers": 1, "lr": 5e-4, "dropout": 0.1, "weight_decay": 1e-5},
        {"hidden_dim": 64, "num_layers": 2, "lr": 5e-4, "dropout": 0.1, "weight_decay": 1e-5},
    ]

    results = []
    for cfg in configs:
        print("Running:", cfg)
        result = run_experiment(cfg, X, Y)
        results.append(result)
        print("Result:", result)

    df = pd.DataFrame(results)
    print(df)
    df.to_csv("experiment_results.csv", index=False)
