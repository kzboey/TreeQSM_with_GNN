import numpy as np

# Unique for each cylinder
class Cylinder:
    def __init__(
        self, 
        branchID=None,
        branchOrder=None,
        cylinderID=None,
        posInBranch=None,
        parentCylID=None,
        childCyID=None,
        startX=None,
        startY=None,
        startZ=None,
        axisX=None,
        axisY=None,
        axisZ=None,
        length=None,
        radius=None,
        angleX=None, # Cylinder angle to x
        angleY=None, # Cylinder angle to y
        angleZ=None, # Cylinder angle to z
        parent=None,
        angle2parent=None, # Cyl angle to parentCyl
        ratio_length2bl=None, # Ratio of cyl length to branch length
        ratio_radius2bl=None, # Ratio of cyl radius to branch length
        volume=None,
        ratio_volume2pv=None, # Ratio of cylinder volume to parent cylinder volume
    ):
        self.branchID = branchID
        self.branchOrder = branchOrder
        self.cylinderID = cylinderID
        self.posInBranch = posInBranch
        self.parentCylID = parentCylID
        self.childCyID = childCyID
        self.startX = startX        
        self.startY = startY        
        self.startZ = startZ   
        self.axisX = axisX        
        self.axisY = axisY        
        self.axisZ = axisZ 
        self.length = length
        self.radius = radius
        self.angleX = self.cyl_angle_to_x()       
        self.angleY = self.cyl_angle_to_y()       
        self.angleZ = self.cyl_angle_to_z()  
        self.parent = parent
        self.angle2parent = self.cyl_angle_to_parent()         
        self.ratio_length2bl = ratio_length2bl        
        self.ratio_radius2bl = ratio_radius2bl 
        self.volume = self.compute_volume()
        self.ratio_volume2pv = self.ratio_volume_2pv()
        
    def cyl_angle_to_x(self):
        magnitude = np.sqrt(self.axisX**2 + self.axisY**2 + self.axisZ**2)
        return np.arccos(self.axisX / magnitude)
    
    def cyl_angle_to_y(self):
        magnitude = np.sqrt(self.axisX**2 + self.axisY**2 + self.axisZ**2)
        return np.arccos(self.axisY / magnitude)

    def cyl_angle_to_z(self):
        magnitude = np.sqrt(self.axisX**2 + self.axisY**2 + self.axisZ**2)
        return np.arccos(self.axisZ / magnitude)

    def cyl_angle_to_parent(self):
        angle = 0
        parentCyl = self.parent
        if parentCyl is not None:
            # Define axis as numpy arrays
            # parent cylinder
            vector_parent = np.array([parentCyl.axisX, parentCyl.axisY, parentCyl.axisZ])
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
            
    # Compute the ratio of cylinder length to branch length
    def ratio_length_2bl(self, branch_length):
        return self.length / branch_length
        
    # Compute the ratio of cylinder radius to branch length
    def ratio_radius_2bl(self, branch_length):
        return self.radius / branch_length
    
    # Compute the volume of each cylinder
    def compute_volume(self):
        return np.pi * self.radius**2 * self.length
    
    # Compute the ratio of cylinder volume to parent cylinder volume
    def ratio_volume_2pv(self):
        parentCyl = self.parent
        if parentCyl is not None:
            return self.volume / parentCyl.volume
        else:
            return 0
    
def getParentCylinder(parentCylID, cylinderArr):
    parent = None
    
    for cyl in cylinderArr:
        if parentCylID == cyl.cylinderID:
            parent = cyl
    return parent 
        
        