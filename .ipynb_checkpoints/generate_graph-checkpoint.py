import os
import pandas as pd
import numpy as np
import networkx as nx
import pickle
import torch
import time

save_path = './pickle/qsm.pickle'
save_path_graph = './pickle/graph_dataset.pickle'
qsm_path = './pickle/object/'
#graph_datalist = []

# Deserialize qsm
# with open(save_path, 'rb') as f:
#     datalist = pickle.load(f) 
    
#for data in datalist:
    
for file_name in os.listdir(qsm_path):
    #print('data')
    if file_name[-6:] != 'pickle':
        continue
    
    start = time.time()
    with open(qsm_path+file_name, 'rb') as f:  
        data = pickle.load(f) 
     
    qsm = data[0] 
    tree_species = data[1] 
    tree_features = qsm['tree']
    branch_features = qsm['branch']
    cylinder_features = qsm['cylinder']

    G = nx.DiGraph()

    tree_dict = vars(tree_features)
    for brch in branch_features:
        branch_dict = vars(brch)
        numCyl = brch.numCylinder
        cutCylinderArr = cylinder_features[brch.startIndex:brch.endIndex]
        for cyl in cutCylinderArr:
            cylinderId = cyl.cylinderID
            parentCylId = cyl.parentCylID
            # merged attributes of tree, branch and cylinder level features
            cylinder_dict = vars(cyl)
            # prepare attributes with torch tensor, put in matrix form
            node_attr_dict = {key: value for d in [tree_dict, branch_dict, cylinder_dict] for key, value in d.items()}
            G.add_node(cylinderId)
            G.nodes[cylinderId].update(node_attr_dict)

            if parentCylId != 0:
                G.add_edge(cylinderId, parentCylId)
    
    # edge_index = torch.tensor(list(G.edges)).t().contiguous()
    data_tuple = (G, tree_species)
    
    treeId = file_name
    save_file = './pickle/graph/{}'.format(file_name)
    with open(save_file, 'wb') as f:  
        pickle.dump(data_tuple, f) # serialize the list
        
    time_delta = time.time() - start
    print('Elapsed time of tree {}: {}'.format(treeId, time_delta))
    #graph_datalist.append(data_tuple)
    
# with open(save_path_graph, 'wb') as f:  
#     pickle.dump(graph_datalist, f) # serialize the list
    

    