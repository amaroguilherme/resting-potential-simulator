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
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, model_type="RNN"):
        """
        RNN-based model for predicting time-dependent membrane potential trajectories.

        Args:
            input_dim (int): Number of input parameters (excluding time).
            hidden_dim (int): Dimension of hidden state in the RNN.
            output_dim (int): Number of output dimensions (e.g., 1 for membrane potential).
            num_layers (int): Number of RNN layers.
            model_type (str): Type of recurrent layer ("RNN", "LSTM", "GRU").
        """
        super(RNNModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.model_type = model_type

        rnn_input_dim = input_dim + 1  

        if model_type == "RNN":
            self.rnn = nn.RNN(rnn_input_dim, hidden_dim, num_layers, batch_first=True)
        elif model_type == "LSTM":
            self.rnn = nn.LSTM(rnn_input_dim, hidden_dim, num_layers, batch_first=True)
        elif model_type == "GRU":
            self.rnn = nn.GRU(rnn_input_dim, hidden_dim, num_layers, batch_first=True)
        else:
            raise ValueError("Invalid model_type. Choose 'RNN', 'LSTM', or 'GRU'.")

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
        batch_size = X.size(0)

        X_seq = X.unsqueeze(1).repeat(1, seq_len, 1)

        t_values = torch.linspace(0, 1, seq_len, device=X.device).unsqueeze(0).unsqueeze(-1)
        t_seq = t_values.repeat(batch_size, 1, 1)

        X_with_time = torch.cat([X_seq, t_seq], dim=-1)

        out, _ = self.rnn(X_with_time)

        Y_pred = self.fc(out).squeeze(-1)

        return Y_pred
    
    
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x_seq):
        out, _ = self.lstm(x_seq)
        last_out = out[:, -1, :]
        y_pred = self.fc(last_out)
        return y_pred.squeeze(-1)
    