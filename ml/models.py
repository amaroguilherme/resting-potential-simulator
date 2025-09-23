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
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.0):
        """
        RNN-based model (LSTM) to predict membrane potential trajectories.

        Args:
            input_dim (int): Number of input features (bio-physical parameters).
            hidden_dim (int): Hidden size of the LSTM.
            output_dim (int): Number of outputs per timestep (usually 1 for V(t)).
            num_layers (int): Number of stacked LSTM layers.
            dropout (float): Dropout rate between LSTM layers.
        """
        super(RNNModel, self).__init__()

        self.encoder = nn.Linear(input_dim, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=1,       # feed a dummy "time signal" at each step
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, seq_len):
        """
        Forward pass of the RNN model.

        Args:
            x (torch.Tensor): Input parameters of shape (batch_size, input_dim).
            seq_len (int): Length of the output trajectory (number of timesteps).

        Returns:
            torch.Tensor: Predicted trajectories of shape (batch_size, seq_len, output_dim).
        """
        batch_size = x.size(0)

        h0 = torch.tanh(self.encoder(x))
        h0 = h0.unsqueeze(0)
        c0 = torch.zeros_like(h0)

        dummy_input = torch.zeros(batch_size, seq_len, 1, device=x.device)

        lstm_out, _ = self.lstm(dummy_input, (h0, c0))

        out = self.decoder(lstm_out)

        return out


def get_model(name, **kwargs):
    """
    Factory method to instantiate models by name.
    """
    if name == "mlp":
        return MLPModel(**kwargs)
    elif name == "rnn":
        return RNNModel(**kwargs)
    else:
        raise ValueError(f"Unknown model: {name}")
