import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ml.datasets import RestingPotentialDataset, load_dataset, split_dataset

from ml.models import MLPModel, RNNModel


SEED = 42
MODEL_NAME = "RNN"        # "MLP" or "RNN"
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
HIDDEN_SIZES = [256, 128]  # to MLP
HIDDEN_DIM = 128            # to RNN
DROPOUT = 0.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("models", exist_ok=True)


def get_model(name, input_dim, output_dim=1):
    if name.lower() == "mlp":
        return MLPModel(input_dim=input_dim, hidden_sizes=HIDDEN_SIZES, output_dim=output_dim)
    elif name.lower() == "rnn":
        return RNNModel(input_dim=input_dim, hidden_dim=HIDDEN_DIM, output_dim=output_dim, dropout=DROPOUT)
    else:
        raise ValueError(f"Unknown model name: {name}")


def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for X_batch, Y_batch in dataloader:
        X_batch = X_batch.float().to(DEVICE)
        Y_batch = Y_batch.float().to(DEVICE)

        optimizer.zero_grad()
        if MODEL_NAME.lower() == "mlp":
            Y_pred = model(X_batch)
        else:  # RNN
            Y_pred = model(X_batch, seq_len=Y_batch.shape[1]).squeeze(-1)
        loss = criterion(Y_pred, Y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, Y_batch in dataloader:
            X_batch = X_batch.float().to(DEVICE)
            Y_batch = Y_batch.float().to(DEVICE)
            if MODEL_NAME.lower() == "mlp":
                Y_pred = model(X_batch)
            else:  # RNN
                Y_pred = model(X_batch, seq_len=Y_batch.shape[1])
            loss = criterion(Y_pred, Y_batch)
            total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(dataloader.dataset)


def main():
    torch.manual_seed(SEED)

    X, Y, _, _ = load_dataset()
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = split_dataset(X, Y)

    input_dim = X_train.shape[1]

    train_dataset = RestingPotentialDataset(X_train, Y_train)
    val_dataset = RestingPotentialDataset(X_val, Y_val)
    test_dataset = RestingPotentialDataset(X_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = get_model(MODEL_NAME, input_dim).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss = evaluate(model, val_loader, criterion)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"models/best_model_{MODEL_NAME}.pt")

    test_loss = evaluate(model, test_loader, criterion)
    print(f"Final Test Loss: {test_loss:.6f}")

    torch.save({
        "model_state": model.state_dict(),
        "input_dim": input_dim,
        "hidden_dim": model.hidden_dim,
        "output_dim": 1,
        "num_layers": model.n_layers,
        "model_type": MODEL_NAME
    }, f"models/final_model_{MODEL_NAME}.pt")

    print("Model saved in 'models/'.")

if __name__ == "__main__":
    main()
