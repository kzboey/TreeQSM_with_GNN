import os
import pickle
import time
import config
import shutil
import numpy as np
import torch
from torch_geometric.data import Data

target_labels = {
    'Acer platanoides' : 0, #
    'Acer pseudoplatanus' : 1, #
    'Aesculus hippocastanum' : 2, #
    'Carpinus betulus' : 3, #
    'Corylus colurna' : 4, #
    'Fraxinus excelsior' : 5, #
    'Platanus × hispanica' : 6, #
    'Populus nigra var. italica' : 7, #
    'Prunus species' : 8, #
    'Robinia pseudoacacia' : 9, #
    'Sorbus aria' : 10, #
    'Tilia cordata' : 11 #
}

target_reverse_labels = {
    0 : 'Acer platanoides', 
    1 : 'Acer pseudoplatanus', 
    2 : 'Aesculus hippocastanum', 
    3 : 'Carpinus betulus', 
    4 : 'Corylus colurna', 
    5 : 'Fraxinus excelsior', 
    6 : 'Platanus × hispanica', 
    7 : 'Populus nigra var. italica', 
    8 : 'Prunus species' , 
    9 : 'Robinia pseudoacacia', 
    10 : 'Sorbus aria', 
    11 : 'Tilia cordata' 
}

target_labels_new = {
    'Acer platanoides' : 0, #
    'Aesculus hippocastanum' : 1, #
    'Corylus colurna' : 2, #
    'Fraxinus excelsior' : 3, #
    'Platanus × hispanica' : 4, #
    'Populus nigra var. italica' : 5, #
    'Robinia pseudoacacia' : 6, #
    'Tilia cordata' : 7 #
}

target_lists = [0,2,4,5,6,7,9,11]

# Get the number of valid samples for each class
count_0 = 0
count_2 = 0
count_4 = 0
count_5 = 0
count_6 = 0
count_7 = 0
count_9 = 0
count_11 = 0

filtered_count = 0

dataset2_path = config.dataset2_path
dataset3_path = config.dataset3_path

_, _, files = os.walk(dataset2_path).__next__()

for file in files:
    data_path = dataset2_path + file
    if os.path.exists(data_path) == False:
        print("{} - not exists".format(data_path))
        continue
    else:
        with open(data_path, 'rb') as f:  
            data = pickle.load(f) 

        # keep only the top 8 species defined in target_labels
        if data.y not in target_lists:
            #os.remove(data_path)
            filtered_count += 1
        else:
            tree_species = target_reverse_labels[data.y.item()]
            data.y = torch.tensor(target_labels_new[tree_species], dtype=torch.long)
            
            new_path = dataset3_path + file
            with open(new_path, 'wb') as f:  
                pickle.dump(data, f) # serialize the list
                        
            if data.y == 0:
                count_0 += 1
            elif data.y == 2:
                count_2 += 1
            elif data.y == 4:
                count_4 += 1                
            elif data.y == 5:
                count_5 += 1
            elif data.y == 6:
                count_6 += 1
            elif data.y == 7:
                count_7 += 1
            elif data.y == 9:
                count_9 += 1
            elif data.y == 11:
                count_11 += 1                
                
print('#Acer platanoides: ',count_0)
print('#Aesculus hippocastanum: ',count_2)
print('#Corylus colurna: ',count_4)
print('#Fraxinus excelsior: ',count_5)
print('#Platanus × hispanica: ',count_6)
print('#Populus nigra var. italica: ',count_7)
print('#Robinia pseudoacacia: ',count_9)
print('#Tilia cordata: ',count_11)
print('#Filtered samples: ',filtered_count) 
print('#Total samples: ',count_0+count_2+count_4+count_5+count_6+count_7+count_9+count_11)       