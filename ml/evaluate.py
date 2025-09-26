import torch
import matplotlib.pyplot as plt
from ml.models import LSTMModel
from ml.datasets import load_dataset, TrajectoryDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_full_trajectory(model, init_seq, steps=500, output_idx=0):
    """
    Predicts a full trajectory autoregressively while keeping input features aligned.

    Args:
        model: trained LSTMModel
        init_seq (torch.Tensor): Initial sequence, shape (1, seq_len, input_dim)
        steps (int): Number of steps to predict
        output_idx (int): Index of the feature that model predicts (usually membrane potential)
    Returns:
        List of predicted values (membrane potential)
    """
    model.eval()
    preds = []

    seq = init_seq.clone()

    for _ in range(steps):
        with torch.no_grad():
            y_pred = model(seq)

        preds.append(y_pred.item())

        next_step = seq[:, -1, :].clone()
        next_step[:, output_idx] = y_pred

        seq = torch.cat([seq[:, 1:, :], next_step.unsqueeze(1)], dim=1)

    return preds


def main():
    X, Y, _, _ = load_dataset()
    seq_len = min(50, X.shape[1])
    dataset = TrajectoryDataset(X, Y, seq_len=seq_len)

    checkpoint = torch.load("models/best_model_lstm.pth", map_location=DEVICE)
    input_dim = checkpoint.get("input_dim", 11)
    hidden_dim = checkpoint.get("hidden_dim", 256)
    num_layers = checkpoint.get("num_layers", 2)

    model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])

    x_input, _ = dataset[0]
    init_seq = x_input.unsqueeze(0).to(DEVICE)

    preds = predict_full_trajectory(model, init_seq, steps=500)

    true_curve = Y[0][:500]

    plt.figure(figsize=(10, 5))
    plt.plot(true_curve, label="Ground Truth")
    plt.plot(preds, label="Prediction", linestyle="--")
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane Potential (mV)")
    plt.legend()
    plt.title("Predicted vs Ground Truth Membrane Potential")
    plt.show()


if __name__ == "__main__":
    main()
