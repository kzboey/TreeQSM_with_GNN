import os
import pickle
import time
import config
import shutil
import numpy as np
import torch
from torch_geometric.data import Data

dataset_path = config.dataset_path
dataset2_path = config.dataset2_path

_, _, files = os.walk(dataset_path).__next__()

for file in files:
    data_path = dataset_path + file
    if os.path.exists(data_path) == False:
        print("{} - not exists".format(data_path))
        continue
    else:
        with open(data_path, 'rb') as f:  
            data = pickle.load(f) 
        
        isNan = torch.isnan(data.x).any().item()
        if isNan:
            print(f'data: {file} contains NaN! (invalid)')
            continue
        else:
            destination = dataset2_path + file
            shutil.copy2(data_path, destination)
        