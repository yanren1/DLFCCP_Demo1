import torch

def mse(y_pred, y_true):
    return torch.mean(torch.abs(y_pred - y_true)**2)