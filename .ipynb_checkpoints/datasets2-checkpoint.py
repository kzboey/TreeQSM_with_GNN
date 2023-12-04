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
    'Aesculus √ó carnea' : 8,
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
    'Platanus √ó hispanic' : 24,
    'Platanus √ó hispanica' : 25,
    'Populus nigra' : 26,
    'Populus nigra var. italica' : 27,
    'Prunus species' : 28,
    'Pyrus species' : 29,
    'Quercus species' : 30,
    'Robinia pseudoacacia' : 31,
    'Sorbus aria' : 32,
    'Taxus baccata' : 33,
    'Tilia cordata' : 34,
    'Tilia species' : 35,
    'Ulmus glabra' : 36,
    'Ulmus laevis' : 37
}

# Dataset that transform qsm object into graph with node features and edge indexes
class QsmDataset(Dataset):
    def __init__(self, dataset):
        self.qsmData = dataset
        self.labels = target_labels
        self.num_features = 72
        #self.batch_size = batch_size
        
    def __len__(self):
        return len(self.qsmData)
        #return (np.ceil(len(self.qsmData) / float(self.batch_size))).astype(np.int)
    
    def __getitem__(self, idx):
        data = self.qsmData[idx]
        qsm = data[0] 
        tree_species = data[1] 
        tree = qsm['tree']
        branch_features = qsm['branch']
        cylinder_features = qsm['cylinder']
        
        first = True
        
        start = time.time()
        print("Start getting dataset ({})".format(tree.treeID))
        for brch in branch_features:
            numCyl = brch.numCylinder
            cutCylinderArr = cylinder_features[brch.startIndex:brch.endIndex]
            for cyl in cutCylinderArr:
                edges = torch.tensor([cyl.cylinderID, cyl.parentCylID])
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
                    float(brch.length),
                    float(brch.radius),
                    float(brch.volume), 
                    float(brch.axisX), 
                    float(brch.axisY), 
                    float(brch.axisZ), 
                    float(brch.angleX),
                    float(brch.angleY), 
                    float(brch.angleZ), 
                    float(brch.angle2parent), 
                    float(brch.ratio_2plength),
                    float(brch.ratio_2pvolume),
                    float(brch.ratio_length2volume),
                    float(brch.ratio_2pradius), 
                    float(brch.avg_dist_child), 
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
                node_attr = torch.vstack((node_attr,node_feature)) if first != True else node_feature
                edge_index = torch.vstack((edge_index,edges)) if first != True else edges                
                first = False
        
        target = torch.tensor(target_labels[tree_species], dtype=torch.long)
        data = Data(x=node_attr, edge_index=torch.transpose(edge_index,0,1), y=target) 
        
        time_delta = time.time() - start
        print('Elapsed time of tree {}: {}'.format(tree.treeID, time_delta))
        
        return data

