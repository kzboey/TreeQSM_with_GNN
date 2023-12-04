import torch
from torch_geometric.loader import DataLoader
from datasets import QsmDataset
from models import GCN, GIN, GraphSAGE, GatedGCN, AGNN, ARMA
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

import numpy as np
from torch.optim import Adam
import time
import wandb
import config
import random
import pickle 
import os
import copy
import math

#################################### RECORD TRAINING ##############################################

wandb.login(key=config.WANDBID)
wandb.init(
    project='TreeML species prediction (GNN model)',
    config={
        "epochs": 100,
        "lr": 1e-3,
    }
)

#################################### RECORD TRAINING ##############################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#################################### READING PARMAETERS ##############################################

# dataset path
save_path_data = config.dataset2_path 
save_path_model = config.model_path #'./models/{}.pth'
        
# GNN PARAM (COMMON)
epochs = config.epochs
n_convs = config.num_convolutional
lr = config.learning_rate
patience = config.patience
batchsize = config.batchsize
decay = config.weight_decay

criterion = torch.nn.CrossEntropyLoss()

#################################### READING PARMAETERS ##############################################

#################################### CREATE DATASET BATCHES ##########################################

filebatches = [] # list of filenames

subdirs, dirs, files = os.walk(save_path_data).__next__()
random.shuffle(files)

# Number of files to train (min: 0, max: total number of file len(files)), batchsize should be less than file_length
batchsize = len(files)  # Max at 128, might cause error if laod more into memory
file_length = len(files)  #len(files) 
filebatches = [files[i:i+batchsize] for i in range(0, file_length, batchsize)] # Save the filenames in in each batches

#################################### CREATE DATASET BATCHES ##########################################


def train(model, loader, model_name):
    optimizer = torch.optim.Adam(model.parameters(),
                              lr=lr,
                              weight_decay=decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.scheduler_decay)
    current_patience = 0
    #model_best = copy.deepcopy(model)
    model.train()

    best_val_loss = float("inf")
    # current_patience = 0
    loss_history = []
    
    start = time.time()
    print("Start training for ({})".format(model_name))
    for epoch in range(epochs):
        train_loss = 0
        acc = 0
        val_loss = 0
        val_acc = 0
        
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)
        
        # Train on batches
        for idx, data in enumerate(tqdm(loader)):
            x, edge_index, batch, targets = data.x.to(device), data.edge_index.to(device), data.batch.to(device), data.y.to(device)
            optimizer.zero_grad()
            _, out = model(x, edge_index, batch)
            loss = criterion(out, targets)
            train_loss += loss / len(loader)
            acc += accuracy(out.argmax(dim=1), targets) / len(loader)
            loss.backward()
            optimizer.step()
            loss_history.append(loss)
            
        # Validation after each epoch
        #scheduler.step()
        val_loss, val_acc, val_f1 = test(model, val_loader, model_name)
        scheduler.step()  # Update learning rate
        
        # Log the loss and accuracy values at the end of each epoch
        wandb.log({
            'Epoch  ({})'.format(model_name) : epoch,
            'Train Loss  ({})'.format(model_name) : train_loss,
            'Train Acc  ({})'.format(model_name) : acc,
            'validation Loss  ({})'.format(model_name) : val_loss,
            'validation Acc  ({})'.format(model_name) : val_acc,
            'validation f1  ({})'.format(model_name) : val_f1,
            'Learning rate ({})'.format(model_name) : scheduler.get_last_lr()})
            
        # Print metrics every 10 epochs
        if(epoch % 10 == 0):
            print(f'Epoch {epoch:>3} | Train Loss: {train_loss:.2f} '
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
        
        #model_best = copy.deepcopy(model)
        model_path = save_path_model.format(model_name)
        torch.save(model, model_path)

    return model

@torch.no_grad()
def test(model, loader, model_name):
    # criterion = torch.nn.CrossEntropyLoss()
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
        metrics = {'Test Loss ({})'.format(model_name): loss, 'Test Accuracy ({})'.format(model_name): acc,
                  'f1 score: ({})'.format(model_name) : f1}
        wandb.log(metrics)
        
    return loss, acc, f1

@torch.no_grad()
def report(model, loader, model_name):
    model.eval()
    
    for i, data in enumerate(tqdm(loader)):
        x, edge_index, batch, targets = data.x.to(device), data.edge_index.to(device), data.batch.to(device), data.y.to(device)
        _, out = model(x, edge_index, batch)
        print(classification_report(targets.to('cpu').numpy(), out.argmax(dim=1).to('cpu').numpy()))
    
def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()
    
#best models to be used on test sets
test_models = []

for idx, filebatch in enumerate(filebatches):
    print(f'--------------- training batch : {idx} -----------------')
    # Create training, validation, and test sets for each batches, 80:10:10 split
    train_dataset = filebatch[:int(len(filebatch)*0.7)]
    val_dataset   = filebatch[int(len(filebatch)*0.7):int(len(filebatch)*0.9)]
    test_dataset  = filebatch[int(len(filebatch)*0.9):]
    
    train_dataset = QsmDataset(save_path_data, train_dataset)
    val_dataset = QsmDataset(save_path_data, val_dataset)
    test_dataset = QsmDataset(save_path_data, test_dataset)

    # Create mini-batches
    train_loader = DataLoader(train_dataset, batch_size=config.batchsize, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batchsize, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batchsize, shuffle=False)
    classification_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    
    save_models = []
    
    if idx == 0:
        num_features = train_dataset.num_features
        num_labels = len(train_dataset.labels)
        
        # Define the 6 GNN model for training
        gcn = GCN(
            in_channel=num_features, 
            n_convs=config.gcn_num_convolutional, 
            dim_h=config.gcn_hidden_dimension, 
            top_K=config.gcn_topkpooling_ratio, 
            out_channel=num_labels)
        
        gin = GIN(
            in_channel=num_features, 
            n_convs=config.gin_num_convolutional, 
            dim_h=config.gin_hidden_dimension, 
            out_channel=num_labels)
        
        gsage = GraphSAGE(
            in_channel=num_features, 
            n_convs=config.gsage_num_convolutional, 
            dim_h=config.gsage_hidden_dimension, 
            top_K=config.gsage_topkpooling_ratio, 
            out_channel=num_labels)
        
        gated = GatedGCN(
            in_channel=num_features, 
            n_convs=config.gated_num_convolutional, 
            dim_h=config.gated_hidden_dimension, 
            top_K=config.gated_topkpooling_ratio, 
            out_channel=num_labels)
        
        agnn = AGNN(
            in_channel=num_features, 
            n_convs=config.agnn_num_convolutional, 
            dim_h=config.agnn_hidden_dimension, 
            top_K=config.agnn_topkpooling_ratio, 
            out_channel=num_labels, 
            num_heads=config.agnn_num_heads)
        
        arma = ARMA(
            in_channel=num_features, 
            n_convs=config.arma_num_convolutional, 
            dim_h=config.arma_hidden_dimension, 
            top_K=config.arma_topkpooling_ratio, 
            out_channel=num_labels)

        #models = [(gcn,'GCN'), (gin,'GIN'), (gsage,"GSage"), (gated,"GatedGCN"), (agnn,"AGNN"), (arma,"ARMA")]
        models = [(gcn,'GCN')]
                  
    for (model, model_name) in models:
        model.to(device)
        best_model = train(model, train_loader, model_name)
        if best_model is not None:
            save_models.append((best_model,model_name))
        else:
            # Early Stopping
            test_models.append((model,model_name))
    
    models = save_models
    
    # last file batch
    if idx == len(filebatches)-1:
        for (model, model_name) in models:
            test_models.append((model, model_name))
            
    torch.cuda.empty_cache()

# Try the best models on test set
print('-' * 20)
for (model, model_name) in test_models:
    test_loss, test_acc, test_f1 = test(model, test_loader, model_name)
    print(f'Test Loss: {test_loss:.2f} | Test Acc: {test_acc*100:.2f}% | Test F1 Acc: {test_f1*100:.2f}%')
    report(model, classification_loader, model_name)