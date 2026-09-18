import numpy as np
import torch
from torch_geometric.data import DataLoader
import os
import sys
sys.path.append("..")
sys.path.append(".")
from utils_file import arg_parser
from model.utils import *
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import InMemoryDataset, Dataset, Data
from itertools import repeat, product, chain

class SimulationDynamic(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(SimulationDynamic, self).__init__(root)
        self.root = root
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')

    def get(self, idx):
        if self.slices:
            data = Data()
            for key in self.data.keys:
                item, slices = self.data[key], self.slices[key]
                s = list(repeat(slice(None), item.dim()))
                s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
                data[key] = item[s]
        else:
            data = self.data
        return data


    def process(self):
        # Read data into huge `Data` list.
        data_list = torch.load(self.raw_paths[0])
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


#         [num_atoms, num_atoms],
#     )


#     data, adj, do, t = batch


if __name__ == '__main__':
    SimulationDynamic('/home/shaoqi/code/SIGN/dataset/HeatDiffusion_10_power_law')


