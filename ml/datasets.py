import numpy as np
import torch

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


class RestingPotentialDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        assert len(X) == len(Y)
        self.X = X
        self.Y = Y
        
        
    def __len__(self):
        return len(self.X)
    
    
    def __getitem__(self, idx):
        X_tensor = torch.tensor(self.X[idx], dtype=torch.float32)
        Y_tensor = torch.tensor(self.Y[idx], dtype=torch.float32)
        
        return X_tensor, Y_tensor
