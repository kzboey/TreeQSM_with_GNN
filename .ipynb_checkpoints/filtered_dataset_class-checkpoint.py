import os
import pickle
import time
import config
import shutil
import numpy as np
import torch
from torch_geometric.data import Data

target_labels = {
    'Acer platanoides' : 3, 
    'Acer pseudoplatanus' : 4, 
    'Aesculus hippocastanum' : 7, 
    'Carpinus betulus' : 12, 
    'Corylus colurna' : 13, 
    'Fraxinus excelsior' : 15, 
    'Platanus × hispanica' : 24, 
    'Populus nigra var. italica' : 26, 
    'Prunus species' : 27, 
    'Robinia pseudoacacia' : 30, 
    'Sorbus aria' : 31, 
    'Tilia cordata' : 33 
}

target_reverse_labels = {
    3 : 'Acer platanoides', 
    4 : 'Acer pseudoplatanus', 
    7 : 'Aesculus hippocastanum', 
    12 : 'Carpinus betulus', 
    13 : 'Corylus colurna', 
    15 : 'Fraxinus excelsior', 
    24 : 'Platanus × hispanica', 
    26 : 'Populus nigra var. italica', 
    27 : 'Prunus species' , 
    30 : 'Robinia pseudoacacia', 
    31 : 'Sorbus aria', 
    33 : 'Tilia cordata' 
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

target_lists = [3,4,7,12,13,15,24,26,27,30,31,33]

# Get the number of valid samples for each class
count_3 = 0
count_4 = 0
count_7 = 0
count_12 = 0
count_13 = 0
count_15 = 0
count_24 = 0
count_26 = 0
count_27 = 0
count_30 = 0
count_31 = 0
count_33 = 0
filtered_count = 0

dataset2_path = config.dataset2_path

_, _, files = os.walk(dataset2_path).__next__()

for file in files:
    data_path = dataset2_path + file
    if os.path.exists(data_path) == False:
        print("{} - not exists".format(data_path))
        continue
    else:
        with open(data_path, 'rb') as f:  
            data = pickle.load(f) 

        # keep only the top 12 species defined in target_labels
        if data.y not in target_lists:
            os.remove(data_path)
            filtered_count += 1
        else:
            tree_species = target_reverse_labels[data.y.item()]
            data.y = torch.tensor(target_labels_new[tree_species], dtype=torch.long)
            
            with open(data_path, 'wb') as f:  
                pickle.dump(data, f) # serialize the list
                
            if data.y == 3:
                count_3 += 1
            elif data.y == 4:
                count_4 += 1
            elif data.y == 7:
                count_7 += 1                
            elif data.y == 12:
                count_12 += 1
            elif data.y == 13:
                count_13 += 1
            elif data.y == 15:
                count_15 += 1
            elif data.y == 24:
                count_24 += 1
            elif data.y == 26:
                count_26 += 1                
            elif data.y == 27:
                count_27 += 1    
            elif data.y == 30:
                count_30 += 1                
            elif data.y == 31:
                count_31 += 1                
            elif data.y == 33:
                count_33 += 1   
                
print('#Acer platanoides: ',count_3)
print('#Acer pseudoplatanus: ',count_4)
print('#Aesculus hippocastanum: ',count_7)
print('#Carpinus betulus: ',count_12)
print('#Corylus colurna: ',count_13)
print('#Fraxinus excelsior: ',count_15)
print('#Platanus × hispanica: ',count_24)
print('#Populus nigra var. italica: ',count_26)
print('#Prunus species: ',count_27)
print('#Robinia pseudoacacia: ',count_30)
print('#Sorbus aria: ',count_31)
print('#Tilia cordata: ',count_33)
        