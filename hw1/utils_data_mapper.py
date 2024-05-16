import numpy as np
from utils_data_reader import get_datasets_pytorch
import json

class StandardScaler():
    def __init__(self, data=None) -> None:
        self.std_min = None
        self.std_max = None
        self.mean = None
        self.std = None
        if (data is not None):            
            self.mean = np.mean(data)
            self.std = np.std(data)
        

    def load_stats(self, fName):
        with open(fName, 'r') as f:
            stats = json.load(f)
        self.mean = stats['mean']
        self.std = stats['std']
        self.std_min = stats['std_min']
        self.std_max = stats['std_max']
        # pretty print like json
        print("Loaded stats:")
        print(f"\tmean={self.mean}\n\tstd={self.std}\n\tstd_min={self.std_min}\n\tstd_max={self.std_max}")
        

    def dump_stats(self):
        return {
            'mean': self.mean,
            'std': self.std,
            'std_min': self.std_min,
            'std_max': self.std_max
        }

    def transform(self, data, update_stats=False):
        data = (data - self.mean) / self.std
        if update_stats:
            self.std_min = np.min(data)
            self.std_max = np.max(data)
        data = (data - self.std_min) / (self.std_max - self.std_min)
        return data
    
    def inverse_transform(self, data):
        data = data * (self.std_max - self.std_min) + self.std_min
        data = data * self.std + self.mean
        return 

if __name__ == "__main__":
    # # Pytorch's Dataset objects. By default, reading in log10
    # train, val, test, original_trainset= get_datasets_pytorch(train_ds=[2011, 2012, 2013, 2014, 2015, 2016], test_ds=[2017], train_ratio=0.8, shuffle=True)

    # ss = StandardScaler(original_trainset)
    # _ = ss.transform(original_trainset)
    # stats = ss.dump_stats()
    # # save as json
    
    # with open('data_stats.json', 'w') as f:
    #     json.dump(stats, f)
    ss = StandardScaler(None)
    ss.load_stats('data_stats.json')

