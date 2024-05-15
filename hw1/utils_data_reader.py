"""
This module provides functions to read the density data and prepare it for training the neural network.
The data is stored in numpy files with the following naming convention: density_{year}.npy
The data is reshaped to the desired dimensions (N, 24, 20, 36) where N is the number of samples.
The data is then split into training, validation, and test sets.
"""

from torch.utils.data import Dataset
import numpy as np

def get_file_name(year:int)->str:
    """
    Returns the file name for the density data for the given year
    """
    assert year in range(2011, 2018), "year should be in range 2011 to 2017"
    return f"data/density_{year}.npy"

def load_data(year:int)->np.ndarray:
    """
    Loads the density data for the given year
    """
    file_name = get_file_name(year)
    x = np.load(file_name)
    return np.log10(x)

def ds_reshaper(raw_data:np.ndarray)->np.ndarray:
    """
    Reshapes the raw data to the desired dimensions (N, 24, 20, 36)
    @ raw_data: the raw data to be reshaped. (172820, N)
    returns: the reshaped data. (N, 24, 20, 36)
    """
    original_shape = raw_data.shape
    N = original_shape[1]
    nofLst = 24
    nofLat = 20
    nofAlt = 36
    
    # Check if the product of the new dimensions matches the original number of rows
    assert original_shape[0] == nofLst * nofLat * nofAlt, "The input data cannot be reshaped to the desired dimensions"
    reshaped_data = np.reshape(raw_data, (nofLst, nofLat, nofAlt, N), order="F")
    # move the last axis to the first
    reshaped_data = np.moveaxis(reshaped_data, -1, 0)
    return reshaped_data

def prep_data_range(years:list[int]):
    """
    Prepares the data for the given range of years. Data is concatenated along the first axis with reshaped. (N, 24, 20, 36)
    @ years: list of years to be used
    returns: the concatenated data for the given years. (N, 24, 20, 36)    
    """
    data = []
    for year in years:
        print(f"Loading data for year {year}")
        data.append(ds_reshaper(load_data(year)))
    print("Concatenating data...")
    return np.concatenate(data, axis=0)

def split_val(data, train_ratio=0.2, shuffle=True, seed=None):
    """
    Splits the data into training and validation sets
    @ data: the data to be split
    @ train_ratio: ratio of the training data to be used for validation
    @ shuffle: whether to shuffle the data before splitting
    @ seed: seed for the random number generator
    returns: the training and validation sets as numpy arrays. (N_train, 24, 20, 36), (N_val, 24, 20, 36)
    """
    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        indices = np.random.permutation(len(data))
    else:
        indices = np.arange(len(data))

    val_size = int(len(data) * (1-train_ratio))
    train_indices = indices[:-val_size]
    val_indices = indices[-val_size:]
    
    train_data = data[train_indices]
    val_data = data[val_indices]
    
    return train_data, val_data

class RhoDataset(Dataset):
    def __init__(self, np_data:np.ndarray=None):
        self.data = np_data
        self.N = self.data.shape[0]

    def shape(self):
        return self.data.shape
    
    def __getitem__(self, i):
        try:
            return self.data[i, :, :, :]
        except:
            print(f"Index {i} out of range")
            return None

    def __len__(self):
        return self.N
    
def move_alt_axis_for_pytorch(data:np.ndarray):
    """
    Moves the altitude axis to the 2nd axis
    @ data: the data to be reshaped. (N, 24, 20, 36)
    returns: the reshaped data. (N, 36, 20, 24)
    """
    x = np.moveaxis(data, 2, 1)
    return np.moveaxis(x, 3, 1)


def move_alt_axis_for_numpy(data:np.ndarray):
    """
    Moves the altitude axis to the 3rd axis
    @ data: the data to be reshaped. (N, 36, 20, 24)
    returns: the reshaped data. (N, 24, 20, 36)
    """
    x = np.moveaxis(data, 1, 3)
    return np.moveaxis(x, 1, 2)

def get_datasets_pytorch(train_ds = [i for i in range(2011, 2017)], test_ds = [2017], train_ratio:float=0.8, shuffle=True)->tuple:
    """
    Prepares the training, validation, and test datasets as pytorch Dataset object
    @ train_ds: list of years to be used for training
    @ test_ds: list of years to be used for testing
    @ train_ratio: ratio of the training data to be used for validation
    @ shuffle: whether to shuffle the data before splitting    
    returns: the training, validation, and test datasets as pytorch Dataset objects
    """
    print("Preparing training data...")
    original_trainset = move_alt_axis_for_pytorch(prep_data_range(train_ds))


    train_dataset, val_dataset = split_val(original_trainset, train_ratio=train_ratio, shuffle=shuffle)
    train_dataset = RhoDataset(np_data=train_dataset)
    val_dataset = RhoDataset(np_data=val_dataset)
    print("Preparing test data...")
    test_data = RhoDataset(np_data=move_alt_axis_for_pytorch(prep_data_range(test_ds)))

    # tabulate the shapes of the datasets with the same number of spaces in the print statement
    print(f"{'Train Set Shape:':<15} {train_dataset.shape()}")
    print(f"{'Val Set Shape:':<15} {val_dataset.shape()}")
    print(f"{'Test Set Shape:':<15} {test_data.shape()}")
    
    return train_dataset, val_dataset, test_data, original_trainset

if __name__ == "__main__":
    #train_ds, val_ds, test_ds = get_datasets_pytorch(train_ratio=0.8, shuffle=True)
    data = prep_data_range([2011, 2012, 2013, 2014, 2015, 2016])
    
    # show mean and std
    print(f"Mean: {np.mean(data):.4f}")
    print(f"Std: {np.std(data):.4f}")
    print(f"Min: {np.min(data):.4f}")
    print(f"Max: {np.max(data):.4f}")

    # standardize the data
    data = (data - np.mean(data)) / np.std(data)
    

    # show mean and std
    print(f"Mean: {np.mean(data):.4f}")
    print(f"Std: {np.std(data):.4f}")
    print(f"Min: {np.min(data):.4f}")
    print(f"Max: {np.max(data):.4f}")



    
    








