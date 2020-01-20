import torch
from torch.utils.data.dataset import Dataset
import pandas as pd
from pathlib import Path


class DataSet(Dataset):
    def __init__(self, file_name, parent_path=None, standardize=False):
        self.path_object = Path()
        self.parent_path = (str(self.path_object.cwd()) if parent_path is None else parent_path)
        self.path = str(self.parent_path) + '/' + str(file_name)
        dataframe = pd.read_csv(self.path)
        dataframe = dataframe.iloc[:, 1:]
        self.outputs = dataframe.iloc[:, 0]
        self.inputs = (dataframe.iloc[:, 1:] if standardize is not True
                       else self.__class__.standardize_data_minmax(dataframe.iloc[:, 1:]))
        self.total_data_len = len(dataframe)

    @staticmethod
    def standardize_data_normal(x):
        x = (x - x.mean()) / x.std()
        x['ones'] = 1
        return x

    @staticmethod
    def standardize_data_minmax(x):
        x = (x - x.min()) / (x.max() - x.min())
        x['ones'] = 1
        return x

    def __getitem__(self, idx):
        return torch.tensor(self.inputs.iloc[idx]).float(), torch.tensor(self.outputs.iloc[idx]).float()

    def __len__(self):
        return len(self.inputs)
