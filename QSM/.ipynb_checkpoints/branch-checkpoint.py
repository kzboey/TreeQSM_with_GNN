import numpy as np

# Unique for each branch
class Branch:
    def __init__(
        self, 
        branchID=None,
        branchOrder=None,
        cylinder1stStartZ=None, # 1st cylinder of branch to calculate avg_dist_child
        length=None,
        radius=None,    # Average of cylinder radius
        volume=None, # Branch volume
        axisX=None,
        axisY=None,
        axisZ=None,
        angleX=None,  # Branch angle to x
        angleY=None,  # Branch angle to y
        angleZ=None,  # Branch angle to z
        parent=None,
        numCylinder=None,
        startIndex=None, # for printing purpose
        endIndex=None, # for printing purpose
        angle2parent=None,  # Branch angle to parent branch
        ratio_2plength=None, #Ratio of branch length to parents Branch length
        ratio_2pvolume=None, # Ratio of branch volume to parent's branch volume
        ratio_length2volume=None, # branch length / branch volume
        ratio_2pradius=None, # Ratio of branch radius to parent's branch radius
        avg_dist_child=None,  # Average distance between child branches, vertical distance between the start points of branches.        
    ):
        self.branchID = branchID
        self.branchOrder = branchOrder
        self.cylinder1stStartZ = cylinder1stStartZ
        self.length = length
        self.radius = radius   
        self.volume = volume 
        self.axisX = axisX        
        self.axisY = axisY   
        self.axisZ = axisZ        
        self.angleX = self.branch_angle_to_x()          
        self.angleY = self.branch_angle_to_y()   
        self.angleZ = self.branch_angle_to_z()  
        self.parent = parent
        self.numCylinder = numCylinder
        self.startIndex = startIndex
        self.endIndex = endIndex
        self.angle2parent = self.branch_angle_to_parent()  
        self.ratio_2plength = self.ratio_2p_length()          
        self.ratio_2pvolume = self.ratio_2p_volume()       
        self.ratio_length2volume = length/volume        
        self.ratio_2pradius = self.ratio_2p_radius() 
        self.avg_dist_child = avg_dist_child
        
        
    def branch_angle_to_x(self):
        magnitude = np.sqrt(self.axisX**2 + self.axisY**2 + self.axisZ**2)
        return np.arccos(self.axisX / magnitude)
    
    def branch_angle_to_y(self):
        magnitude = np.sqrt(self.axisX**2 + self.axisY**2 + self.axisZ**2)
        return np.arccos(self.axisY / magnitude)

    def branch_angle_to_z(self):
        magnitude = np.sqrt(self.axisX**2 + self.axisY**2 + self.axisZ**2)
        return np.arccos(self.axisZ / magnitude)  
    
    def branch_angle_to_parent(self): 
        angle = 0
        parentBch = self.parent
        if parentBch is not None:
            # Define axis as numpy arrays
            # parent branch
            vector_parent = np.array([parentBch.axisX, parentBch.axisY, parentBch.axisZ])
            # current cylinder
            vector_this = np.array([self.axisX, self.axisY, self.axisZ]) 
            
            # Calculate the dot product of the two vectors
            dot_product = np.dot(vector_parent, vector_this)
            
            # Calculate the magnitudes (lengths) of the vectors
            magnitude_parent = np.linalg.norm(vector_parent)
            magnitude_this = np.linalg.norm(vector_this)
            
            # Calculate the angle in radians using the dot product and magnitudes
            cosine_theta = dot_product / (magnitude_parent * magnitude_this)
            angle = np.arccos(cosine_theta)     
            
        return angle            
    
    def ratio_2p_length(self):
        parentBrch = self.parent
        if parentBrch is not None:
            return self.length / parentBrch.length
        else:
            return 0

    def ratio_2p_volume(self):
        parentBrch = self.parent
        if parentBrch is not None:
            return self.volume / parentBrch.volume
        else:
            return 0

    def ratio_2p_radius(self):
        parentBrch = self.parent
        if parentBrch is not None:
            return self.radius / parentBrch.radius
        else:
            return 0
        
    def compute_avg_dist_child(self, cylinder1stBranchArr, numBranch):
        total_vertical_distance = 0.0
        for cyl in cylinder1stBranchArr:
            if self.branchID != cyl.branchID:
                total_vertical_distance += abs(self.cylinder1stStartZ - cyl.startZ)
                
        return total_vertical_distance/numBranch-1        
        
def getParentBranch(branchArr, parentBranchId, parentBranchOrder):
    parentBranch = None
    # print('pass parentBranch id : ',parentBranchId)
    # print('pass parentBranch order: ',parentBranchOrder)    
    for branch in branchArr:
        # print('parentBranch id : ',branch.branchID)
        # print('parentBranch order: ',branch.branchOrder)
        if (branch.branchID == parentBranchId) and (branch.branchOrder == parentBranchOrder):
            parentBranch = branch
    return parentBranch  
