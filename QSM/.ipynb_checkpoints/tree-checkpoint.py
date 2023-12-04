import numpy as np

# numerical trick to avoid 0 division
eps = 1e-6

# Unique for each tree
class Tree:
    def __init__(
        self, 
        projectID=None,
        treeID=None, 
        dbh=None,
        tree_height=None, # th
        crown_start_height=None, # csh
        crown_projection_area=None, # crown projection area
        max_crown_diameter=None,
        tree_volume=None,
        # cr_height_max=None,   # crown radius in 72 direction, each 5 degrees at max height
        # cr_height_02m=None,   # crown radius in 72 direction, each 5 degrees at height of 2m
        # cr_height_04m=None,   # crown radius in 72 direction, each 5 degrees at height of 4m
        # cr_height_06m=None,   # crown radius in 72 direction, each 5 degrees at height of 6m
        # cr_height_08m=None,   # crown radius in 72 direction, each 5 degrees at height of 8m
        # cr_height_10m=None,   # crown radius in 72 direction, each 5 degrees at height of 10m
        # cr_height_12m=None,   # crown radius in 72 direction, each 5 degrees at height of 12m
        # cr_height_14m=None,   # crown radius in 72 direction, each 5 degrees at height of 14m
        # cr_height_16m=None,   # crown radius in 72 direction, each 5 degrees at height of 16m
        # cr_height_18m=None,   # crown radius in 72 direction, each 5 degrees at height of 18m
        # cr_height_20m=None,   # crown radius in 72 direction, each 5 degrees at height of 20m
        lcl=None, # Largest Crown length (LCL)
        ratio_branch2height=None,     # Ratio of average of 1st branch length to tree height
        ratio_CL_dim=None,   # Ratio between LCL and Max crown diameter
        # avg_crown_radius=None, # Average max crown radius
        # avg_crown_radius_02m=None,
        # avg_crown_radius_04m=None,
        # avg_crown_radius_06m=None,
        # avg_crown_radius_08m=None,
        # avg_crown_radius_10m=None,
        # avg_crown_radius_12m=None,
        # avg_crown_radius_14m=None,
        # avg_crown_radius_16m=None,
        # avg_crown_radius_18m=None,
        # avg_crown_radius_20m=None,
        ratio_dbh_th=None,  # DBH/TH;
        ratio_dbh_volume=None,  # DBH/Tree volume;
        ratio_dbh_minsrad=None,  # DBH/minimum stem radius
        stem_length=None, # temporarily also crown start height
        stem_taper=None,
        stem_volume=None,
        ratio_slength_theight=None, # Stem length / tree height
        axisX=None,
        axisY=None,
        axisZ=None,
        angleX=None, # Angle between stem direction vector and axis x
        angleY=None, # Angle between stem direction vector and axis y
        angleZ=None, # Angle between stem direction vector and axis z
        volume8010=None,      # Tree volume within 80-90% of the tree height / tree volume within 10% of the tree height
        ratio_maxcdim_height=None, # Ratio between Max crown diameter and TH
        ratio_clength_volume=None, # Total cylinder length/tree volume
        ratio_volume_area=None, # Tree volume / crown cover area; 
        ratio_volume_height=None, # Tree volume / tree height;
        ratio_volume_diameter=None, # Tree volume / crown diameter 
        ratio_csh_th=None, # Ratio between crown start height and tree height
        cd_xy=None,
        ccd_xy=None,
        cd_xz=None,       
        ccd_xz=None,  
        cd_yz=None,      
        ccd_yz=None,
        ratio_lcl_cdxy=None, # Ratio between LCL and crown diameter that derived from projected crown points on the x-y plane
        ratio_lcl_th=None, # Ratio between LCL and tree height
        ratio_cdxy_th=None, # Ratio between CD_xy and tree height
        ratio_cdxy_ccdxy=None,
        ratio_cdxz_ccdxz=None,
        ratio_cdyz_ccdyz=None,
    ):
        self.projectID = projectID
        self.treeID = treeID
        self.dbh = dbh  
        self.tree_height = tree_height  
        self.crown_start_height = crown_start_height 
        self.crown_projection_area = crown_projection_area
        self.max_crown_diameter = max_crown_diameter
        self.tree_volume = tree_volume  
        # self.cr_height_max = cr_height_max
        # self.cr_height_02m = cr_height_02m
        # self.cr_height_04m = cr_height_04m
        # self.cr_height_06m = cr_height_06m
        # self.cr_height_08m = cr_height_08m
        # self.cr_height_10m = cr_height_10m
        # self.cr_height_12m = cr_height_12m
        # self.cr_height_14m = cr_height_14m
        # self.cr_height_16m = cr_height_16m
        # self.cr_height_18m = cr_height_18m
        # self.cr_height_20m = cr_height_20m
        self.lcl = self.tree_height - self.crown_start_height
        self.ratio_branch2height = ratio_branch2height   
        self.ratio_CL_dim = self.lcl / self.max_crown_diameter
        # self.avg_crown_radius = sum(self.cr_height_max) / 72
        # self.avg_crown_radius_02m = sum(self.cr_height_02m) / 72
        # self.avg_crown_radius_04m = sum(self.cr_height_04m) / 72
        # self.avg_crown_radius_06m = sum(self.cr_height_06m) / 72
        # self.avg_crown_radius_08m = sum(self.cr_height_08m) / 72
        # self.avg_crown_radius_10m = sum(self.cr_height_10m) / 72
        # self.avg_crown_radius_12m = sum(self.cr_height_12m) / 72
        # self.avg_crown_radius_14m = sum(self.cr_height_14m) / 72
        # self.avg_crown_radius_16m = sum(self.cr_height_16m) / 72
        # self.avg_crown_radius_18m = sum(self.cr_height_18m) / 72
        # self.avg_crown_radius_20m = sum(self.cr_height_20m) / 72
        self.ratio_dbh_th = self.dbh / self.tree_height   
        self.ratio_dbh_volume = self.dbh / self.tree_volume       
        self.ratio_dbh_minsrad = ratio_dbh_minsrad
        self.stem_length = crown_start_height # temporary
        self.stem_taper = stem_taper
        self.stem_volume = stem_volume
        self.ratio_slength_theight = self.stem_length / self.tree_height 
        self.axisX = axisX
        self.axisY = axisY
        self.axisZ = axisZ
        self.angleX = angleX # self.compute_angleX()        
        self.angleY = angleY # self.compute_angleY()        
        self.angleZ = angleZ # self.compute_angleZ()  
        self.volume8010 = volume8010
        self.ratio_maxcdim_height = self.max_crown_diameter / self.tree_height
        self.ratio_clength_volume = ratio_clength_volume
        self.ratio_volume_area = self.tree_volume / self.crown_projection_area  
        self.ratio_volume_height = self.tree_volume / self.tree_height
        self.ratio_volume_diameter = self.tree_volume / self.max_crown_diameter       
        self.ratio_csh_th = self.crown_start_height  / self.tree_height 
        self.cd_xy = cd_xy        
        self.ccd_xy = ccd_xy  
        self.cd_xz = cd_xz        
        self.ccd_xz = ccd_xz  
        self.cd_yz = cd_yz        
        self.ccd_yz = ccd_yz  
        self.ratio_lcl_cdxy = self.lcl / self.cd_xy 
        self.ratio_lcl_th = self.lcl / self.tree_height 
        self.ratio_cdxy_th = self.cd_xy / self.tree_height  
        self.ratio_cdxy_ccdxy = self.cd_xy / self.ccd_xy
        self.ratio_cdxz_ccdxz = self.cd_xz / self.ccd_xz 
        self.ratio_cdyz_ccdyz = self.cd_yz / self.ccd_yz        
    
    def compute_branch2height(self, branchArr):
        total_length_1stbranch = 0 # 1st branch length
        for branch in branchArr:
            if branch.branchOrder == 1:
                total_length_1stbranch += branch.length
        return total_length_1stbranch
    
    # compute the ration of DBH/minimum stem radius
    def compute_ratio_dbh_minsrad(self, cylinderArr):
        min_stem_radius = 0.0
        stem_length = 0.0
        for cyl in cylinderArr:
            if cyl.branchOrder == 0 and stem_length < self.stem_length:
                stem_length += cyl.length
                min_stem_radius = cyl.radius if cyl.radius < min_stem_radius else min_stem_radius
            elif stem_length > self.tree_height:
                break
        return min_stem_radius
    
    def compute_stem_taper(self, cylinderArr):
        rsb = 0.0 # lower stem radius
        rst = 0.0 # upper stem radius
        index_rst = 0
        stem_length = 0.0
        
        for index, cyl in enumerate(cylinderArr):
            if cyl.branchOrder == 0 and stem_length < self.stem_length:
                stem_length += cyl.length
                if cyl.cylinderID == 1:
                    rsb = cyl.radius   
            elif stem_length > self.tree_height:
                rst = cylinderArr[index-1].radius
                break
                
        return (rsb-rst) / self.stem_length
    
    def compute_stem_volume(self, cylinderArr):
        stem_length = 0.0
        stem_volume = 0.0
        
        for cyl in cylinderArr:
            if cyl.branchOrder == 0 and stem_length < self.stem_length:
                stem_length += cyl.length
                stem_volume += cyl.volume
            elif stem_length > self.tree_height:
                break
        
        return stem_volume
    
    def compute_axisX(self, cylinderArr):
        stem_length = 0.0
        stem_sumUx = 0.0
        stem_sumUy = 0.0
        stem_sumUz = 0.0
        
        for cyl in cylinderArr:
            if cyl.branchOrder == 0 and stem_length < self.stem_length:
                stem_length += cyl.length
                stem_sumUx +=  cyl.axisX*cyl.length
                stem_sumUy +=  cyl.axisX*cyl.length
                stem_sumUz +=  cyl.axisX*cyl.length
        
        magnitude = np.sqrt(stem_sumUx**2 + stem_sumUy**2 + stem_sumUz**2)
        return stem_sumUx/magnitude

    def compute_axisY(self, cylinderArr):
        stem_length = 0.0
        stem_sumUx = 0.0
        stem_sumUy = 0.0
        stem_sumUz = 0.0
        
        for cyl in cylinderArr:
            if cyl.branchOrder == 0 and stem_length < self.stem_length:
                stem_length += cyl.length
                stem_sumUx +=  cyl.axisX*cyl.length
                stem_sumUy +=  cyl.axisX*cyl.length
                stem_sumUz +=  cyl.axisX*cyl.length
        
        magnitude = np.sqrt(stem_sumUx**2 + stem_sumUy**2 + stem_sumUz**2)
        return stem_sumUy/magnitude

    def compute_axisZ(self, cylinderArr):
        stem_length = 0.0
        stem_sumUx = 0.0
        stem_sumUy = 0.0
        stem_sumUz = 0.0
        
        for cyl in cylinderArr:
            if cyl.branchOrder == 0 and stem_length < self.stem_length:
                stem_length += cyl.length
                stem_sumUx +=  cyl.axisX*cyl.length
                stem_sumUy +=  cyl.axisX*cyl.length
                stem_sumUz +=  cyl.axisX*cyl.length
        
        magnitude = np.sqrt(stem_sumUx**2 + stem_sumUy**2 + stem_sumUz**2)
        return stem_sumUz/magnitude
    
    def compute_angleX(self):
        magnitude = np.sqrt(self.axisX**2 + self.axisY**2 + self.axisZ**2)
        return np.arccos(self.axisX / magnitude)

    def compute_angleY(self):
        magnitude = np.sqrt(self.axisX**2 + self.axisY**2 + self.axisZ**2)
        return np.arccos(self.axisY / magnitude)

    def compute_angleZ(self):
        magnitude = np.sqrt(self.axisX**2 + self.axisY**2 + self.axisZ**2)
        return np.arccos(self.axisZ / magnitude)

    def compute_volume8010(self, cylinderArr):
        volume10 = 0.0 # volume at 10% of tree height
        volume80 = 0.0 # volume at 80% of tree height
        lower_startZ = 0.0  # lowest z coordinate value of any cylinder of tree
        uppper_startZ = 0.0 # highest z coordinate value of any cylinder of tree
        
        for cyl in cylinderArr:
            lower_startZ = cyl.startZ if cyl.startZ < lower_startZ else lower_startZ
            uppper_startZ = cyl.startZ if cyl.startZ > uppper_startZ else uppper_startZ
            
        axisZ_10limit = lower_startZ + 0.1*(uppper_startZ-lower_startZ)  # z coordinate value at 10% height
        axisZ_80limit = lower_startZ + 0.8*(uppper_startZ-lower_startZ)  # z coordinate value at 80% height
        
        for cyl in cylinderArr:
            if cyl.startZ < axisZ_10limit:
                volume10 += cyl.volume
            
            if cyl.startZ < axisZ_80limit:
                volume80 += cyl.volume
                
        return volume80/volume10
    
    def compute_ratio_clength_volume(self, cylinderArr):
        total_cyl_length = 0.0
        
        for cyl in cylinderArr:
            total_cyl_length += cyl.length
            
        return total_cyl_length / self.tree_volume
    
    def compute_ccd_xy(self):
        pass