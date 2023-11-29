import torch
from torch.utils.data import Dataset
import pandas as pd
import random
class PacketDataSet(Dataset):
    def __init__(
            self, path, train = True):
        self.train = train
        if isinstance(path, list):
            dataframes = [pd.read_csv(i, sep=',') for i in path]
            self.df = pd.concat(dataframes, ignore_index=True)
        else:
            self.df = pd.read_csv(path, sep=',')
        dfs = [self.df]

        # Łączenie wszystkich DataFrame'ów w jeden, powielając wiersze
        self.df = pd.concat(dfs, ignore_index=True)
        self.rows = len(self.df)
        self.number_list = list(range(4, self.rows))  # This creates a list of numbers from 1 to 10




        random.shuffle(self.number_list)
        if self.train:
            self.number_list = self.number_list[int(0.2*self.rows):]
        else:
            self.number_list = self.number_list[:int(0.2 * self.rows)]
        self.df = self.df.astype('float32')
        #print(self.df.columns)
        #print(self.df.isna().sum())
        self.x = torch.tensor(self.df.values, dtype=torch.float32)

        x1 = self.x[0].unsqueeze(0)
        x2 = self.x[1].unsqueeze(0)
        x3 = self.x[2].unsqueeze(0)
        x4 = self.x[3].unsqueeze(0)
        x5 = self.x[4].unsqueeze(0)
        summed = torch.cat([x1, x2, x3, x4, x5], dim=0)
        summed = self.normalize_column(summed, 0)
        summed = self.normalize_column(summed, 1)
        summed = self.normalize_column(summed, 2)
        summed = self.normalize_column(summed, 3)

        print(summed)



    def get_columns(self):
        return(list(self.df.columns.values))
    def __len__(self):
        return len(self.number_list)

    def normalize_column(self, tensor, column_index):
        """Normalizuje wybraną kolumnę tensora przy użyciu normalizacji min-max."""
        min_val = torch.min(tensor[:, column_index])
        max_val = torch.max(tensor[:, column_index])
        if max_val - min_val != 0:
            tensor[:, column_index] = (tensor[:, column_index] - min_val) / (max_val - min_val)
        else:
            # Możesz zdecydować co zrobić, gdy min i max są takie same
            # Na przykład, ustaw wszystkie wartości na 0 lub zachowaj je bez zmian
            tensor[:, column_index] = 0  # lub zachowaj bez zmian

        return tensor

    def __getitem__(self, index):
        index_from_random = self.number_list[index]
        x1 = self.x[index_from_random-4].unsqueeze(0)
        x2 = self.x[index_from_random-3].unsqueeze(0)
        x3 = self.x[index_from_random-2].unsqueeze(0)
        x4 = self.x[index_from_random-1].unsqueeze(0)
        x5 = self.x[index_from_random].unsqueeze(0)
        summed = torch.cat([x1,x2,x3,x4,x5],dim = 0)
        summed = self.normalize_column(summed, 0)
        summed = self.normalize_column(summed, 1)
        summed = self.normalize_column(summed, 2)
        summed = self.normalize_column(summed, 3)

        #print(summed)

        return summed

