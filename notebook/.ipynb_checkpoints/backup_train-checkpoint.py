#import networkx as nx
import torch
from torch_geometric.loader import DataLoader
from datasets import QsmDataset
from models import GCN, GIN, GraphSAGE, GatedGCN, AGNN, ARMA
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import numpy as np
from torch.optim import Adam
import time
import wandb
import config
import random
import pickle 
import os
import copy

# wandb.login(key=config.WANDBID)
# wandb.init(
#     project='TreeML species prediction (GNN model)',
#     config={
#         "epochs": 100,
#         "lr": 0.001,
#     }
# )

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# dataset path
save_path_qsm = config.QSM_PATH 
save_path_model = config.MODEL_PATH #'./models/{}.pth'
        
filebatches = [] # list of filenames
batchsize = 500 # Read 500 files into memory for training

subdirs, dirs, files = os.walk(save_path_qsm).__next__()
random.shuffle(files)
file_length = 100 #len(files) # Number of files to train (min: 0, max: total number of file len(files))
filebatches = [files[i:i+batchsize] for i in range(0, file_length, batchsize)] # Save the filenames in in each batches

# GNN PARAM
n_convs = config.NUM_CONVOLUTIONAL
dim_h = config.HIDDEN_DIMENSION
top_K = config.TOPKPOOLING_RATIO

epochs = 3 
patience = 10

def train(model, loader, model_name):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=0.001,
                                  weight_decay=0.01)
    model_best = copy.deepcopy(model)
    model.train()
    total_loss = 0
    acc = 0
    val_loss = 0
    val_acc = 0
    best_val_loss = 0
    
    start = time.time()
    print("Start training for ({})".format(model_name))
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)
        
        # Train on batches
        for idx, data in enumerate(tqdm(loader)):
            x, edge_index, batch, targets = data.x.to(device), data.edge_index.to(device), data.batch.to(device), data.y.to(device)
            optimizer.zero_grad()
            _, out = model(x, edge_index, batch)
            loss = criterion(out, targets)
            total_loss += loss / len(loader)
            acc += accuracy(out.argmax(dim=1), targets) / len(loader)
            loss.backward()
            optimizer.step()
            
        # Validation after each epoch
        val_loss, val_acc, val_f1 = test(model, val_loader, model_name)

        #Log    
        # metrics = {'Training Loss ({})'.format(model_name): total_loss, 'Training Accuracy ({})'.format(model_name): accuracy}
        # wandb.log(metrics)
            
        # Print metrics every 10 epochs
        if(epoch % 10 == 0):
            print(f'Epoch {epoch:>3} | Train Loss: {total_loss:.2f} '
                  f'| Train Acc: {acc*100:>5.2f}% '
                  f'| Val Loss: {val_loss:.2f} '
                  f'| Val Acc: {val_acc*100:.2f}%'
                  f'| Val F1 Acc: {val_f1*100:.2f}%')
        
        # Check for improvement in validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            current_patience = 0
        else:
            current_patience += 1

        # Check if early stopping should be triggered
        if current_patience >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            return None
        
        model_best = copy.deepcopy(model)
        model_path = save_path_model.format(model_name)
        torch.save(model, model_path)
        
    test_loss, test_acc, test_f1 = test(model_best, test_loader, model_name)
    print(f'Test Loss: {test_loss:.2f} | Test Acc: {test_acc*100:.2f}% | Test F1 Acc: {test_f1*100:.2f}%')

    return model_best

@torch.no_grad()
def test(model, loader, model_name):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    loss = 0
    acc = 0
    
    # Train on batches 
    for i, data in enumerate(tqdm(loader)):
        x, edge_index, batch, targets = data.x.to(device), data.edge_index.to(device), data.batch.to(device), data.y.to(device)
        _, out = model(x, edge_index, batch)
        loss += criterion(out, targets) / len(loader)
        acc += accuracy(out.argmax(dim=1), targets) / len(loader)
        f1 = f1_score(targets.cpu(),out.argmax(dim=1).cpu(), average="weighted")
        #Log    
        # metrics = {'Test Loss ({})'.format(model_name): loss, 'Test Accuracy ({})'.format(model_name): acc,
        #           'f1 score: ({})'.format(model_name) : f1}
        # wandb.log(metrics)
        
    return loss, acc, f1
        
def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()
    

for idx, filebatch in enumerate(filebatches):
    print(f'---------- training batch : {idx} ------------')
    # Create training, validation, and test sets for each batches, 80:10:10 split
    train_dataset = filebatch[:int(len(filebatch)*0.8)]
    val_dataset   = filebatch[int(len(filebatch)*0.8):int(len(filebatch)*0.9)]
    test_dataset  = filebatch[int(len(filebatch)*0.9):]
    
    train_dataset = QsmDataset(save_path_qsm, train_dataset)
    val_dataset = QsmDataset(save_path_qsm, val_dataset)
    test_dataset = QsmDataset(save_path_qsm, test_dataset)

    # Create mini-batches
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    save_models = []
    
    if idx == 0:
        num_features = train_dataset.num_features
        num_labels = len(train_dataset.labels)
        
        # Define the 6 GNN model for training
        gcn = GCN(in_channel=num_features, n_convs=n_convs, dim_h=dim_h, top_K=top_K, out_channel=num_labels)
        gin = GIN(in_channel=num_features, n_convs=n_convs, dim_h=dim_h, out_channel=num_labels)
        gsage = GraphSAGE(in_channel=num_features, n_convs=n_convs, dim_h=dim_h, top_K=top_K, out_channel=num_labels)
        gated = GatedGCN(in_channel=num_features, n_convs=n_convs, dim_h=128, top_K=top_K, out_channel=num_labels)
        agnn = AGNN(in_channel=num_features, n_convs=n_convs, dim_h=dim_h, top_K=top_K, out_channel=num_labels, num_heads=8)
        arma = ARMA(in_channel=num_features, n_convs=n_convs, dim_h=dim_h, top_K=top_K, out_channel=num_labels)

        models = [(gcn,'GCN'), (gin,'GIN'), (gsage,"GSage"), (gated,"GatedGCN"), (agnn,"AGNN"), (arma,"ARMA")]
        
    for (model, model_name) in models:
        model.to(device)
        best_model = train(model, train_loader, model_name)
        save_models.append((best_model,model_name))
    
    models = save_models