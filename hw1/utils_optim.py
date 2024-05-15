import torch
import torch.nn as nn
import numpy as np

class CustomLoss(nn.Module):
    def __init__(self, alpha=0.9, epsillon=1e-6):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.epsillon = epsillon    
        self.mse_loss = nn.MSELoss()

    def update_alpha(self, alpha):
        self.alpha = alpha

    def forward(self, y_true, y_pred, ):
        # Mean Squared Error (MSE)
        mse = self.mse_loss(y_pred, y_true)
        
        # Mean Absolute Percentage Error (MAPE)
        mape = torch.mean(torch.abs((y_true - y_pred) / (y_true + self.epsillon)))

        # print(f"MSE: {mse}, MAPE: {mape}")
        
        # Combined loss
        loss = self.alpha * mse + (1 - self.alpha) * mape
        return loss
    
class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), f'./weights/autoencoder.pth')


def numpy_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def numpy_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6)))

def numpy_custom_loss(y_true, y_pred, alpha=0.9):
    mse = numpy_mse(y_true, y_pred)
    mape = numpy_mape(y_true, y_pred)
    total = alpha * mse + (1 - alpha) * mape
    return mse, mape, total

