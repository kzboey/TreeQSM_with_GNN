import os
import pandas as pd
import numpy as np
import csv
import sys
from QSM.cylinder import Cylinder, getParentCylinder
from QSM.branch import Branch, getParentBranch
from QSM.tree import Tree
from QSM import utils
import time
import pickle
import config
    

# filepath_tree ='./Data/TreeML_Dataset.csv'
filepath_tree = config.TREE_DATASET_LOCAL
tree_dataset = pd.read_csv(filepath_tree)

save_path = './pickle/qsm.pickle'

#datalist = []
# reading previous data
# if os.path.exists(save_path):
#     with open(save_path, 'rb') as f:
#         tmp_datalist = pickle.load(f) 
#     lastItem = tmp_datalist[-1]
#     lastItemId = lastItem[0]['tree'].treeID
#     print('last tree in datalist: ',lastItemId)
#     numItems = len(tmp_datalist)
#     if numItems > 0:
#         datalist = tmp_datalist
#         index_start = tree_dataset.index[tree_dataset['treeID']==lastItemId].tolist()[0]+1
#         index_stop = len(tree_dataset)
#         tree_dataset = tree_dataset[index_start:index_stop]


for idx, (index, rowQsm) in enumerate(tree_dataset.iterrows()):
    start = time.time()
    valid = True
    qsm = {'tree': None, 'branch': None, 'cylinder': None}
    
    filepath_opt_csv = utils.get_optqsm_path(rowQsm['projectID'], rowQsm['treeID'])
    
    if filepath_opt_csv is None:
        print('Dataset: {} not found'.format(rowQsm['treeID']))
        continue
        
    qsm_dataset = pd.read_csv(filepath_opt_csv)
    condition = qsm_dataset['branchID']
    
    valid = False if all(item == 0 for item in rowQsm[15:87].values) else True
    if rowQsm['botanical_name'] == '?':
        valid = False
    
    if qsm_dataset is not None and valid:        
        
        try:
            unique_branchId = qsm_dataset['branchID'].unique().tolist() # Obtain the each distinct branch IDs
            branchArr = [] # Array for all branches of each tree
            cylinderArr = [] # Array for all cylinder of each branch in tree
            cylinder1stBranchArr = [] # Array for first cylinder only of each branch in tree

            for bid in unique_branchId:
                # Initialize branch properties
                numCylinder = (condition==bid).sum()
                #print('branch IDs : {}, #Cylinders : {}'.format(bid,numCylinder))
                b_length = 0
                b_radius = 0
                b_volume = 0
                b_sumUx, b_sumUy, b_sumUz = 0, 0, 0
                cylinder1st = None
                bid_1stcylinder_parent = -1 # branch id of the parent of the 1st cylinder of branch
                border_1stcylinder_parent = -1 # branch order of the parent of the 1st cylinder of branch

                itemNum = 1
                # loop over all cylider for respective branch'ids
                for index, row in qsm_dataset[qsm_dataset['branchID']==bid].iterrows():                  

                    parentCyl = getParentCylinder(row['parentCylID'], cylinderArr)
                    parentBranch = None

                    cylinder = Cylinder(
                        branchID=row['branchID'],
                        branchOrder=row['branchOrder'],
                        cylinderID=row['cylinderID'],
                        posInBranch=row['posInBranch'],
                        parentCylID=row['parentCylID'],
                        childCyID=row['childCyID'],
                        startX=row['start_x'],
                        startY=row['start_y'],
                        startZ=row['start_z'],
                        axisX=row['axis_x'],
                        axisY=row['axis_y'],
                        axisZ=row['axis_z'],
                        length=row['length'],
                        radius=row['radius'],
                        parent=parentCyl
                    )
                    cylinderArr.append(cylinder)

                    if itemNum == 1:
                        cylinder1st = cylinder
                        cylinder1stBranchArr.append(cylinder)

                    # Updating branch properties
                    b_length += row['length']
                    b_radius += row['radius']/numCylinder
                    b_volume += cylinder.volume
                    b_sumUx +=  row['axis_x']*row['length']
                    b_sumUy +=  row['axis_y']*row['length']
                    b_sumUz +=  row['axis_z']*row['length']

                    # get parent branch
                    # if index == 0 and parentCyl is not None:
                    #     parentBranch = getParentBranch(branchArr, parentCyl.branchID, parentCyl.branchOrder)

                    # initalize branch at last cylinder
                    if itemNum == numCylinder:
                        # print('Item Number: ',itemNum)
                        if len(branchArr) != 0:
                            parentBranch = branchArr[-1]

                        b_magnitude = np.sqrt(b_sumUx**2 + b_sumUy**2 + b_sumUz**2)
                        # Initialize branch object
                        branch = Branch(
                            branchID=row['branchID'],
                            branchOrder=row['branchOrder'],
                            cylinder1stStartZ=cylinder1st.startZ,
                            length=b_length,
                            radius=b_radius,
                            volume=b_volume,
                            axisX=b_sumUx/b_magnitude,     
                            axisY=b_sumUy/b_magnitude,
                            axisZ=b_sumUz/b_magnitude,
                            parent=parentBranch,      
                        ) 
                        branchArr.append(branch)
                        # Complete cylinder object that require branch properties
                        # for cy in cylinderArr:
                        #     #print('cy branch IDs : {}, branch id : {}'.format(cy.branchID,branch.branchID))
                        #     cy.ratio_length2bl=cy.ratio_length_2bl(branch.length)
                        #     cy.ratio_radius2bl=cy.ratio_radius_2bl(branch.length)                                                               
                    itemNum += 1

            # update vertical distance between branches
            numBranch = len(branchArr)
            for branch in branchArr:
                branch.avg_dist_child = branch.compute_avg_dist_child(cylinder1stBranchArr,numBranch)
                cylinder_of_branch = [cyl for cyl in cylinderArr if cyl.branchID == branch.branchID]
                for cyl in cylinder_of_branch:
                    cyl.ratio_length2bl=cyl.ratio_length_2bl(branch.length)
                    cyl.ratio_radius2bl=cyl.ratio_radius_2bl(branch.length)   

            # Initialize tree object
            tree = Tree(
                projectID=rowQsm['projectID'],
                treeID=rowQsm['treeID'],
                dbh=rowQsm['DBH_m_'],
                tree_height=rowQsm['treeHeight_m_'],
                crown_start_height=rowQsm['crownStartHeight_m_'],
                crown_projection_area=rowQsm['crownProjectionArea_m2_'],
                max_crown_diameter=rowQsm['crownDiameterMax_m_'],
                tree_volume=rowQsm['totalVolume_L_'],
                cr_height_max=rowQsm[15:87].values,
                cr_height_02m=rowQsm[87:159].values,
                cr_height_04m=rowQsm[159:231].values,
                cr_height_06m=rowQsm[231:303].values,
                cr_height_08m=rowQsm[303:375].values,
                cr_height_10m=rowQsm[375:447].values,
                cr_height_12m=rowQsm[447:519].values,
                cr_height_14m=rowQsm[519:591].values,
                cr_height_16m=rowQsm[591:663].values,
                cr_height_18m=rowQsm[663:735].values,
                cr_height_20m=rowQsm[735:807].values,
                cd_xy=rowQsm['crownN_45d_m_']+rowQsm['crownS_45d_m_'],
                ccd_xy=rowQsm['crownW_45d_m_']+rowQsm['crownE_45d_m_'],
                cd_xz=rowQsm['treeHeight_m_'],
                ccd_xz=rowQsm['crownN_45d_m_']+rowQsm['crownS_45d_m_'],
                cd_yz=rowQsm['treeHeight_m_'],
                ccd_yz=rowQsm['crownW_45d_m_']+rowQsm['crownE_45d_m_'],
            )
            tree.ratio_branch2height = tree.compute_branch2height(branchArr)
            tree.ratio_dbh_minsrad = tree.compute_ratio_dbh_minsrad(cylinderArr)
            tree.stem_taper = tree.compute_stem_taper(cylinderArr)
            tree.stem_volume = tree.compute_stem_volume(cylinderArr)
            tree.ratio_clength_volume = tree.compute_ratio_clength_volume(cylinderArr)
            tree.axisX = tree.compute_axisX(cylinderArr)
            tree.axisY = tree.compute_axisY(cylinderArr)
            tree.axisZ = tree.compute_axisZ(cylinderArr)
            tree.angleX = tree.compute_angleX()
            tree.angleY = tree.compute_angleY()
            tree.angleZ = tree.compute_angleZ()
            tree.volume8010 = tree.compute_volume8010(cylinderArr)

            # Update qsm parameter
            qsm['tree']=tree
            qsm['branch']=branchArr
            qsm['cylinder']=cylinderArr

            tree_species = rowQsm['botanical_name']
            data_tuple = (qsm, tree_species)
            #datalist.append(data_tuple)
            
            save_file = './pickle/object/{}_{}.pickle'.format(idx,rowQsm['treeID'])
            with open(save_file, 'wb') as f:  
                pickle.dump(data_tuple, f) # serialize the list
            
            # save to pickle every n iteration
            # if index%100 == 0:
            #     print('saving for dataset ',index)
            #     with open(save_path, 'wb') as f:  
            #         pickle.dump(datalist, f) # serialize the list
                
        except ZeroDivisionError as e:
            print(print('tree {} : {}'.format(rowQsm['treeID'],e)))
            continue
    else:
        print('tree {} is invalid'.format(rowQsm['treeID']))
        continue
    time_delta = time.time() - start
    print('Elapsed time of tree {}: {}'.format(rowQsm['treeID'], time_delta))

# Serialize qsm
# with open(save_path, 'wb') as f:  
#     pickle.dump(datalist, f) # serialize the list
 