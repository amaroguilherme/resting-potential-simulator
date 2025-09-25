import numpy as np
import torch

from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

def load_dataset():
    data = np.load('dataset/files/dataset.npz')
    X = data['X']
    Y = data['Y']
    t = data['t']
    param_names = data['param_names']
    
    return X, Y, t, param_names


def split_dataset(X, Y, train_frac=0.7, val_frac=0.15, seed=42):
    np.random.seed(seed)
    N = len(X)
    indices = np.random.permutation(N)

    n_train = int(N * train_frac)
    n_val = int(N * val_frac)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)


def get_dataloader(dataset, batch_size=64, shuffle=True):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle)


class TrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, seq_len=5):
        """
        Args:
            X (np.ndarray): shape (num_trajectories, traj_len)
            Y (np.ndarray): same shape as X
            seq_len (int): window size
        """
        self.sequences = []
        self.targets = []
        self.seq_len = seq_len

        for traj in range(X.shape[0]):
            traj_X = X[traj]
            traj_Y = Y[traj]
            for i in range(len(traj_X) - seq_len):
                self.sequences.append(traj_X[i:i+seq_len])
                self.targets.append(traj_Y[i+seq_len])

        self.sequences = torch.tensor(self.sequences, dtype=torch.float32).unsqueeze(-1)
        self.targets = torch.tensor(self.targets, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]
