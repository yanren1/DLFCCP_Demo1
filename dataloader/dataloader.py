import os
import pandas as pd
import torch
from torch.utils.data import Dataset


class SampleDataset(Dataset):
    def __init__(self, root_dir, file_name):
        super(SampleDataset, self).__init__()

        self.root_dir = root_dir
        self.file_name = file_name
        self.samples = self.__read_xlsx()


    def __getitem__(self, index):
        samples = self.samples[index]
        # sample, target = samples[:-3],samples[-3:]

        return samples[:-1], samples[-1]

    def __len__(self):
        return len(self.samples)

    def __read_xlsx(self):
        f_pth = os.path.join(self.root_dir, self.file_name)
        # f_pth = os.path.join(root_dir, 'data.xlsx')
        df = pd.read_csv(f_pth,)
        samples = torch.from_numpy(df.to_numpy()).float()

        return samples