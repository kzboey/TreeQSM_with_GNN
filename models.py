import torch 
import torch.nn as nn
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, GatedGraphConv, GATv2Conv, ARMAConv, TopKPooling, global_mean_pool, global_add_pool, BatchNorm
import torch.nn.functional as F

# Suggested models from Chattoraj's paper

# Create Graph Convolutional Network (GCN) model
class GCN(nn.Module):
    def __init__(self, in_channel, n_convs, dim_h, top_K, out_channel):
        super(GCN, self).__init__()
        self.n_convs = n_convs
        self.gcn_layers = nn.ModuleList()
        self.batch_layer = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        
        for i in range(self.n_convs):
            self.gcn_layers.append(GCNConv(in_channel, dim_h))
            self.batch_layer.append(BatchNorm(dim_h))
            in_channel = dim_h
            
            # No top K pooling for last conv layer
            if i!= self.n_convs-1:
                self.pool_layers.append(TopKPooling(dim_h, top_K))
                
        self.lin1 = Linear(dim_h, dim_h)    
        self.lin2 = Linear(dim_h, out_channel)
        
    def forward(self, x, edge_index, batch=None):
        # Node embeddings 
        for i in range(self.n_convs):
            x = self.gcn_layers[i](x, edge_index)
            x = self.batch_layer[i](x)
            x = x.relu()
            
            if i != self.n_convs-1:
                x, edge_index, _, batch, _, _ = self.pool_layers[i](x, edge_index, None, batch)
            
        # Graph-level readout
        hG = global_mean_pool(x, batch)
        
        # Classifier
        # h = F.dropout(hG, p=0.5, training=self.training)
        # h = self.lin(h)
        h = self.lin1(hG)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        
        return F.log_softmax(h, dim=1)
    
# Create Graph Convolutional Network (GCN) with skip connection model
class GCNSkip(nn.Module):
    def __init__(self, in_channel, n_convs, dim_h, top_K, out_channel):
        super(GCN, self).__init__()
        self.n_convs = n_convs
        self.gcn_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        for i in range(self.n_convs):
            self.gcn_layers.append(GCNConv(in_channel, dim_h))
            in_channel = dim_h
            
            # No top K pooling for last conv layer
            if i!= self.n_convs-1:
                self.pool_layers.append(TopKPooling(dim_h, top_K))
                
        self.lin1 = Linear(dim_h, dim_h)    
        self.lin2 = Linear(dim_h, out_channel)   
        
    def forward(self, x, edge_index, batch):
        # Node embeddings 
        for i in range(self.n_convs):
            x = self.gcn_layers[i](x, edge_index)
            x = x.relu()
            
            if i != self.n_convs-1:
                x, edge_index, _, batch, _, _ = self.pool_layers[i](x, edge_index, None, batch)
            
        # Graph-level readout
        hG = global_mean_pool(x, batch)
        
        # Classifier
        h = F.dropout(hG, p=0.5, training=self.training)
        h = self.lin(h)
        
        return hG, F.log_softmax(h, dim=1)

    
# Create Graph Isomorphism Network (GIN) model
class GIN(nn.Module):
    def __init__(self, in_channel, n_convs, dim_h, out_channel):
        super(GIN, self).__init__()
        self.n_convs = n_convs
        self.gin_layers = nn.ModuleList()
        for _ in range(self.n_convs):
            self.gin_layers.append(
                GINConv(
                    Sequential(Linear(in_channel, dim_h),
                    BatchNorm1d(dim_h), ReLU(),
                    Linear(dim_h, dim_h), ReLU()))
            )
            in_channel = dim_h
        self.lin1 = Linear(dim_h*n_convs, dim_h*n_convs)
        self.bn1 = BatchNorm1d(dim_h*n_convs)
        self.lin2 = Linear(dim_h*n_convs, out_channel)
    
    def forward(self, x, edge_index, batch):
        h_s = []
        # Node embeddings
        for i in range(self.n_convs):
            h_s.append(
                self.gin_layers[i](x, edge_index)
            )
            x = h_s[-1] 
        
        # Graph-level readout
        for i in range(self.n_convs):
            h_s[i] = global_add_pool(h_s[i], batch)  # (*, 32)
         
        # Concatenate graph embeddings
        h = torch.cat(h_s, dim=1)
                
        # Classifier
        h = self.lin1(h)
        h = self.bn1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
                
        return h, F.log_softmax(h, dim=1)
            
    
# Create Graph Sample and Aggregated Embeddings (GraphSAGE) model
class GraphSAGE(nn.Module):
    def __init__(self, in_channel, n_convs, dim_h, top_K, out_channel):
        super(GraphSAGE, self).__init__()
        self.n_convs = n_convs
        self.gsage_layers = nn.ModuleList()
        self.batch_layer = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        for i in range(self.n_convs):
            self.gsage_layers.append(SAGEConv(in_channel, dim_h))
            self.batch_layer.append(BatchNorm(dim_h))
            in_channel = dim_h
            
            # No top K pooling for last conv layer
            if i!= self.n_convs-1:
                self.pool_layers.append(TopKPooling(dim_h, top_K))
                
        self.lin1 = Linear(dim_h, dim_h)    
        self.lin2 = Linear(dim_h, out_channel)
    
    def forward(self, x, edge_index, batch):
        # Node embeddings 
        for i in range(self.n_convs):
            x = self.gsage_layers[i](x, edge_index)
            x = self.batch_layer[i](x)
            x = x.relu()
            
            if i != self.n_convs-1:
                x, edge_index, _, batch, _, _ = self.pool_layers[i](x, edge_index, None, batch)
            
        # Graph-level readout
        hG = global_mean_pool(x, batch)
        
        # Classifier
        h = self.lin1(hG)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        
        return hG, F.log_softmax(h, dim=1)
    
# Create a gated graph convolutional layer (GatedGcn),
class GatedGCN(nn.Module):
    def __init__(self, in_channel, n_convs, dim_h, top_K, out_channel):
        super(GatedGCN, self).__init__()           
        self.n_convs = n_convs
        self.gated_layers = nn.ModuleList()
        self.batch_layer = nn.ModuleList()
        self.pool_layers = nn.ModuleList() 
        for i in range(self.n_convs):
            self.gated_layers.append(GatedGraphConv(out_channels=dim_h, num_layers=n_convs))
            self.batch_layer.append(BatchNorm(dim_h))
            # No top K pooling for last conv layer
            if i!= self.n_convs-1:
                self.pool_layers.append(TopKPooling(dim_h, top_K))
        
        self.lin = Linear(dim_h, out_channel) 
        # self.lin1 = Linear(dim_h, dim_h)    
        # self.lin2 = Linear(dim_h, out_channel)     

    def forward(self, x, edge_index, batch=None):
        # Node embeddings 
        for i in range(self.n_convs):
            x = self.gated_layers[i](x, edge_index)
            x = self.batch_layer[i](x)
            x = x.relu()
            
            if i != self.n_convs-1:
                x, edge_index, _, batch, _, _ = self.pool_layers[i](x, edge_index, None, batch)
            
        # Graph-level readout
        hG = global_mean_pool(x, batch)
        
        # Classifier
        # h = self.lin1(hG)
        # h = h.relu()
        # h = F.dropout(h, p=0.5, training=self.training)
        # h = self.lin2(h)
        h = F.dropout(hG, p=0.5, training=self.training)
        h = self.lin(h)
        
        return F.log_softmax(h, dim=1)

# Create a attention based graph neural network (AGNN)
class AGNN(nn.Module):
    def __init__(self, in_channel, n_convs, dim_h, top_K, out_channel, num_heads=8):
        super(AGNN, self).__init__()
        self.n_convs = n_convs
        self.agnn_layers = nn.ModuleList()
        self.batch_layer = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        for i in range(self.n_convs):
            if i != self.n_convs-1:
                self.agnn_layers.append(GATv2Conv(in_channel, dim_h, heads=num_heads))
                self.batch_layer.append(BatchNorm(dim_h*num_heads))
            else:
                # last layer
                self.agnn_layers.append(GATv2Conv(in_channel, dim_h, heads=1))
                self.batch_layer.append(BatchNorm(dim_h))
            in_channel = dim_h*num_heads
            
            # No top K pooling for last conv layer
            if i!= self.n_convs-1:
                self.pool_layers.append(TopKPooling(dim_h*num_heads, top_K)) 
                
        self.lin1 = Linear(dim_h, dim_h)    
        self.lin2 = Linear(dim_h, out_channel) 
                
    def forward(self, x, edge_index, batch):
        # Node embeddings 
        for i in range(self.n_convs):
            x = self.agnn_layers[i](x, edge_index)
            x = self.batch_layer[i](x)
            x = x.relu()
            
            if i != self.n_convs-1:
                x, edge_index, _, batch, _, _ = self.pool_layers[i](x, edge_index, None, batch)
            
        # Graph-level readout
        hG = global_mean_pool(x, batch)
        
        # Classifier
        h = self.lin1(hG)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        
        return hG, F.log_softmax(h, dim=1)

# Create a auto-regressive moving average convolutional layer (ARMA) 
class ARMA(nn.Module):
    def __init__(self, in_channel, n_convs, dim_h, top_K, out_channel):
        super(ARMA, self).__init__()
        self.n_convs = n_convs
        self.arma_layers = nn.ModuleList()
        self.batch_layer = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        for i in range(self.n_convs):
            self.arma_layers.append(ARMAConv(in_channel, dim_h))
            self.batch_layer.append(BatchNorm(dim_h))
            in_channel = dim_h
            
            # No top K pooling for last conv layer
            if i!= self.n_convs-1:
                self.pool_layers.append(TopKPooling(dim_h, top_K))
                
        self.lin1 = Linear(dim_h, dim_h)    
        self.lin2 = Linear(dim_h, out_channel) 
    
    def forward(self, x, edge_index, batch):
        # Node embeddings 
        for i in range(self.n_convs):
            x = self.arma_layers[i](x, edge_index)
            x = self.batch_layer[i](x)
            x = x.relu()
            
            if i != self.n_convs-1:
                x, edge_index, _, batch, _, _ = self.pool_layers[i](x, edge_index, None, batch)
            
        # Graph-level readout
        hG = global_mean_pool(x, batch)
        
        # Classifier
        h = self.lin1(hG)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        
        return hG, F.log_softmax(h, dim=1)
    
    
    

