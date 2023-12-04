## PATH ##
TREE_DATASET_SERVER = './Data/TreeML_Dataset.csv' 
TREE_DATASET_LOCAL = '/Users/boeykaizhe/Documents/TUM/QSM/TreeML_Dataset.csv'
qsm_path = './pickle/object/'
qsm_path2 = './pickle/obj2/'
dataset_path = './datasets/' 
dataset2_path = './datasets2/' # Filtered dataset (12 species)
dataset3_path = './datasets3/' # Filtered dataset (8 species)
model_path = './models/{}.pth' # path to save model

## WEIGHT & BIASES ##
training_log = True  # set to False if don't want to use weight and biase 
WANDBID = '4ae0c463491edacd5223d5b6dbf93207a3996db8'

## GNN PARAMETER (COMMON) ##
epochs = 80
num_convolutional = 3
learning_rate = 1e-3
patience = 50
batchsize = 64
weight_decay = 1e-3 # regularization
step_size = 5 # decay every n epoch
scheduler_decay = 0.9 # to decat set <1i

## GCN PARAMETER ##
gcn_num_convolutional = 3
gcn_hidden_dimension = 512
gcn_topkpooling_ratio = 0.8

## GIN PARAMETER ##
gin_num_convolutional = 3
gin_hidden_dimension = 256

## GSAGE PARAMETER ##
gsage_num_convolutional = 3
gsage_hidden_dimension = 256
gsage_topkpooling_ratio = 0.8

## GatedGCN PARAMETER ##
gated_num_convolutional =2
gated_hidden_dimension = 128
gated_topkpooling_ratio = 0.8

## AGNN PARAMETER ##
agnn_num_convolutional = 2
agnn_hidden_dimension = 128
agnn_topkpooling_ratio = 0.8
agnn_num_heads = 3

## ARMA PARAMETER ##
# step_size=5, decay=0.9, epoch=80
arma_num_convolutional = 3
arma_hidden_dimension = 256
arma_topkpooling_ratio = 0.8

## Classification report ##
# target_names = ['Acer platanoides', 'Acer pseudoplatanus', 'Aesculus hippocastanum', 'Carpinus betulus','Corylus colurna', 'Fraxinus excelsior', 'Platanus × hispanica','Populus nigra var. italica', 'Prunus species', 'Robinia pseudoacacia', 'Sorbus aria', 'Tilia cordata']
target_names = ['Acer platanoides', 'Aesculus hippocastanum', 'Corylus colurna', 'Fraxinus excelsior', 'Platanus × hispanica','Populus nigra var. italica', 'Robinia pseudoacacia', 'Tilia cordata']
