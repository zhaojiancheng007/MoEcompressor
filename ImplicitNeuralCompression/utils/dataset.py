from torch.utils.data import Dataset


class Mydataset(Dataset):
    def __init__(self, coordinates):
        self.coordinates = coordinates
        
    def __getitem__(self, idx):
        data = self.coordinates[idx, :]
        return data
    
    def __len__(self):
        return self.coordinates.shape[0]