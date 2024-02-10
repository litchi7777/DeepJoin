import numpy as np
from torch.utils.data import Dataset
from utils.data_processing import *
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, data):
        data = np.array(data)
        self.table_names = data[:,0]
        self.column_names = data[:,1]
        self.column_values = data[:,2]

    def __len__(self):
        return len(self.table_names)

    def __getitem__(self, idx):
        return self.table_names[idx], self.column_names[idx], self.column_values[idx]
    
def setup_dataloaders(batch_size=100):
    train_data = get_column_data_from_folder("./datasets/raw/train")
    val_data = get_column_data_from_folder("./datasets/raw/val")
    test_data = get_column_data_from_folder("./datasets/raw/test")

    train_dataset = CustomDataset(train_data)
    val_dataset = CustomDataset(val_data)
    test_dataset = CustomDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader, test_loader