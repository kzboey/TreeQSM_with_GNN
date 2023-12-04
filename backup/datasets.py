import os
import pickle
import time

import numpy as np
import torch
from torch.utils.data import Dataset 
from torch_geometric.data import Data,InMemoryDataset, download_url

target_labels = {
    'Acer campestre' : 0, 
    'Acer japonicum' : 1,
    'Acer negundo' : 2, 
    'Acer platanoides' : 3,
    'Acer pseudoplatanus' : 4,
    'Acer saccharinum' : 5, 
    'Acer species' : 6, 
    'Aesculus hippocastanum' : 7,
    'Aesculus × carnea' : 8,
    'Ailanthus altissima' : 9,
    'Betula pendula' : 10,
    'Betula utilis' : 11,
    'Carpinus betulus' : 12,
    'Corylus colurna' : 13,
    'Fagus sylvatica' : 14,
    'Fraxinus excelsior' : 15,
    'Fraxinus ornus' : 16,
    'Ginkgo biloba' : 17,
    'Gleditsia triacanthos' : 18,
    'Liriodendron tulipifera' : 19,
    'Magnolia species' : 20,
    'Malus species' : 21,
    'Picea abies' : 22,
    'Picea species' : 23,
    'Platanus × hispanica' : 24,
    'Populus nigra' : 25,
    'Populus nigra var. italica' : 26,
    'Prunus species' : 27,
    'Pyrus species' : 28,
    'Quercus species' : 29,
    'Robinia pseudoacacia' : 30,
    'Sorbus aria' : 31,
    'Taxus baccata' : 32,
    'Tilia cordata' : 33,
    'Tilia species' : 34,
    'Ulmus glabra' : 35,
    'Ulmus laevis' : 36
}

# Dataset that transform qsm object into graph with node features and edge indexes
class QsmDataset(Dataset):
    def __init__(self, datapath, dataset):
        self.datapath = datapath
        self.qsmData = dataset # list of filenames
        self.labels = target_labels
        self.num_features = 72
        
    def __len__(self):
        return len(self.qsmData)
    
    def __getitem__(self, idx):
        data_path = self.datapath + self.qsmData[idx]
        if os.path.exists(data_path) == False:
            print("{} - not exists".format(data_path))
        else:
            with open(data_path, 'rb') as f:  
                data = pickle.load(f) 
 
            return data

    

    

