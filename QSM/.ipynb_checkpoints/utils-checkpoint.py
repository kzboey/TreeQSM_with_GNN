import os

# get corresponding optqsm file from project and tree id
def get_optqsm_path(projectId, treeId):
    directory_path = '/Users/boeykaizhe/Documents/TUM/QSM/Data/Dataset_QSM'
    #directory_path = './Data/Dataset_QSM'
    
    file_path = '{}/{}/optcsv/OptQSM_{}.csv'.format(directory_path, projectId, treeId)
    
    if os.path.exists(file_path):
        return file_path
    else:
        print('Cannot find file {}'.format(file_path))
        return None
    