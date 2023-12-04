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
    #'Aesculus √ó carnea' : 8,
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
    # 'Platanus √ó hispanica' : 24,
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
            start = time.time()
            with open(data_path, 'rb') as f:  
                data = pickle.load(f) 
                
            qsm = data[0] 
            tree_species = data[1] 

            tree = qsm['tree']
            branches = qsm['branch']
            cylinders = qsm['cylinder']

            first_node = True
            first_edge = True
            edge_list = []

            for cyl in cylinders:
                branch = next((branch for branch in branches if branch.branchID == cyl.branchID), None)
                node_feature = torch.tensor([
                    float(tree.ratio_branch2height),
                    float(tree.ratio_CL_dim),
                    float(tree.avg_crown_radius),
                    float(tree.dbh),
                    float(tree.ratio_dbh_th),
                    float(tree.ratio_dbh_volume),
                    float(tree.ratio_dbh_minsrad),
                    float(tree.stem_taper),
                    float(tree.stem_length),
                    float(tree.ratio_slength_theight),
                    float(tree.axisX),
                    float(tree.axisY),
                    float(tree.axisZ),
                    float(tree.angleX),
                    float(tree.angleY),
                    float(tree.angleZ),
                    float(tree.stem_volume),
                    float(tree.volume8010),
                    float(tree.ratio_maxcdim_height),
                    float(tree.ratio_clength_volume),
                    float(tree.ratio_volume_area),
                    float(tree.ratio_volume_height),
                    float(tree.ratio_volume_diameter),
                    float(tree.crown_start_height),
                    float(tree.ratio_csh_th),
                    float(tree.lcl),
                    float(tree.ratio_lcl_th),
                    float(tree.cd_xy),
                    float(tree.ccd_xy),
                    float(tree.cd_xz),
                    float(tree.ccd_xz),
                    float(tree.cd_yz),
                    float(tree.ccd_yz),
                    float(tree.ratio_lcl_cdxy),
                    float(tree.ratio_cdxy_th),
                    float(tree.ratio_cdxy_ccdxy),
                    float(tree.ratio_cdxz_ccdxz),
                    float(tree.ratio_cdyz_ccdyz),
                    float(branch.length),
                    float(branch.radius),
                    float(branch.volume), 
                    float(branch.axisX), 
                    float(branch.axisY), 
                    float(branch.axisZ), 
                    float(branch.angleX),
                    float(branch.angleY), 
                    float(branch.angleZ), 
                    float(branch.angle2parent), 
                    float(branch.ratio_2plength),
                    float(branch.ratio_2pvolume),
                    float(branch.ratio_length2volume),
                    float(branch.ratio_2pradius), 
                    float(branch.avg_dist_child), 
                    int(cyl.cylinderID), 
                    int(cyl.posInBranch), 
                    int(cyl.parentCylID), 
                    int(cyl.childCyID), 
                    float(cyl.startX), 
                    float(cyl.startY), 
                    float(cyl.startZ), 
                    float(cyl.axisX), 
                    float(cyl.axisY),
                    float(cyl.axisZ),
                    float(cyl.radius), 
                    float(cyl.angleX), 
                    float(cyl.angleY), 
                    float(cyl.angleZ),
                    float(cyl.angle2parent),
                    float(cyl.ratio_length2bl), 
                    float(cyl.ratio_radius2bl), 
                    float(cyl.volume), 
                    float(cyl.ratio_volume2pv) 
                ])
                node_attr = torch.vstack((node_attr,node_feature)) if first_node != True else node_feature
                first_node = False

                if cyl.parentCylID != 0:
                    edges = [cyl.parentCylID, cyl.parentCylID]
                    edge_list.append(edges)
                    first_edge = False

            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            target = torch.tensor(target_labels[tree_species], dtype=torch.long)
            data = Data(x=node_attr, edge_index= edge_index, y=target) 
            time_delta = time.time() - start
            print('Elapsed time of tree {}: {}'.format(tree.treeID, time_delta))
            
            return data

    

    

