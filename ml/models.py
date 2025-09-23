import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """
    Base class for all models. 
    Provides interface and enforces implementation of forward().
    """
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class MLPModel(BaseModel):
    """
    Baseline feedforward neural network (MLP).
    Maps input parameters X -> membrane potential trajectory V(t).
    """
    def __init__(self, input_dim, output_dim, hidden_sizes=[256, 128], dropout=0.0):
        super().__init__()

        layers = []
        prev_size = input_dim

        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_size = h

        layers.append(nn.Linear(prev_size, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (batch, input_dim) array of biophysical parameters
        Returns:
            V_pred: (batch, output_dim) predicted trajectory
        """
        return self.net(x)


class RNNModel(nn.Module):
    """
    RNN-based model (LSTM) to predict membrane potential trajectories.

    Args:
        input_dim (int): Number of input features (bio-physical parameters).
        hidden_dim (int): Hidden size of the LSTM.
        output_dim (int): Number of outputs per timestep (usually 1 for V(t)).
        num_layers (int): Number of stacked LSTM layers.
        dropout (float): Dropout rate between LSTM layers.
    """
    def __init__(self, input_dim, hidden_dim, output_dim=1, n_layers=1, dropout=0.0):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)


    def forward(self, X, seq_len):
        """
        Forward pass of the RNN model.

        Args:
            X (torch.Tensor): Input parameters of shape (batch_size, input_dim).
            seq_len (int): Length of the output trajectory (number of timesteps).

        Returns:
            torch.Tensor: Predicted trajectories of shape (batch_size, seq_len).
        """

        X_seq = X.unsqueeze(1).repeat(1, seq_len, 1)

        out, _ = self.rnn(X_seq)

        Y_pred = self.fc(out).squeeze(-1)

        return Y_pred

