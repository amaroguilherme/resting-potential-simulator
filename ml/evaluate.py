import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

from ml.datasets import RestingPotentialDataset, load_dataset, split_dataset
from ml.models import MLPModel, RNNModel


def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluate the model on a given dataloader (validation or test).

    Args:
        model: Trained model.
        dataloader: DataLoader containing X and Y.
        criterion: Loss function (e.g., nn.MSELoss).
        device: Device to run evaluation (cpu or cuda).

    Returns:
        (mse, rmse, r2): Evaluation metrics.
    """
    model.eval()
    mse_losses = []
    y_true_all, y_pred_all = [], []

    with torch.no_grad():
        for X_batch, Y_batch in dataloader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            Y_pred = model(X_batch, seq_len=Y_batch.shape[1])

            loss = criterion(Y_pred, Y_batch)
            mse_losses.append(loss.item())

            # collect predictions for R²
            y_true_all.append(Y_batch.cpu().numpy())
            y_pred_all.append(Y_pred.cpu().numpy())

    y_true_all = np.concatenate(y_true_all, axis=0)
    y_pred_all = np.concatenate(y_pred_all, axis=0)

    mse = np.mean(mse_losses)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_all.flatten(), y_pred_all.flatten())

    return mse, rmse, r2


def plot_predictions(model, dataset, t, device, n_samples=5):
    """
    Plot model predictions compared to ground truth trajectories.

    Args:
        model: Trained model.
        dataset: Test dataset.
        t: Time vector (ms).
        device: Device to run evaluation (cpu or cuda).
        n_samples: Number of samples to plot.
    """
    model.eval()
    indices = np.random.choice(len(dataset), size=n_samples, replace=False)

    plt.figure(figsize=(12, 8))
    for i, idx in enumerate(indices, 1):
        X, Y_true = dataset[idx]
        X = X.unsqueeze(0).to(device)  # add batch dimension
        Y_true = Y_true.numpy()

        with torch.no_grad():
            Y_pred = model(X, seq_len=Y_true.shape[0])
            Y_pred = Y_pred.squeeze(0).cpu().numpy()

        plt.subplot(n_samples, 1, i)
        plt.plot(t, Y_true, label="Ground Truth", color="black")
        plt.plot(t, Y_pred, label="Prediction", color="red", linestyle="--")
        plt.ylabel("Membrane Potential (mV)")
        if i == 1:
            plt.legend()
        if i == n_samples:
            plt.xlabel("Time (ms)")

    plt.suptitle("Comparison: Ground Truth vs. Predicted Trajectories")
    plt.tight_layout()
    plt.show()


def main():
    model_path = "models/final_model_rnn.pt"
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, Y, t, _ = load_dataset()
    (_, _), (_, _), (X_test, Y_test) = split_dataset(X, Y)

    test_dataset = RestingPotentialDataset(X_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    checkpoint = torch.load(model_path, map_location=device)
    if checkpoint["model_type"] == "RNN":
        model = RNNModel(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            output_dim=checkpoint["output_dim"],
            n_layers=checkpoint["num_layers"]
        )
    elif checkpoint["model_type"] == "MLP":
        model = MLPModel(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            output_dim=checkpoint["output_dim"]
        )
    else:
        raise ValueError(f"Unknown model type: {checkpoint['model_type']}")
    
    input_dim = X.shape[1]
    output_dim = 1

    # Choose the same model architecture as used during training
    model = RNNModel(input_dim=input_dim, hidden_dim=128, output_dim=output_dim, n_layers=1)

    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    # Quantitative evaluation
    criterion = nn.MSELoss()
    mse, rmse, r2 = evaluate_model(model, test_loader, criterion, device)
    print(f"Test set results:")
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R²:   {r2:.4f}")

    # Qualitative evaluation (plots)
    plot_predictions(model, test_dataset, t, device, n_samples=5)


if __name__ == "__main__":
    main()
