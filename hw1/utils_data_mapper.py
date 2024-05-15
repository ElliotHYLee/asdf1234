import numpy as np

class StandardScaler():
    def __init__(self, data) -> None:
        self.mean = np.mean(data)
        self.std = np.std(data)
    
    def transform(self, data):
        data = (data - self.mean) / self.std
        self.min = np.min(data)
        self.max = np.max(data)
        data = (data - self.min) / (self.max - self.min)
        return data
    
    def inverse_transform(self, data):
        data = data * (self.max - self.min) + self.min
        data = data * self.std + self.mean
        return 
    

