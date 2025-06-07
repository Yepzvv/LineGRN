import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch.autograd import Variable
from VGCN import *
file_dir = os.path.dirname(os.path.realpath('__file__'))
    
class Net(torch.nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_size, latent_dim=[128, 128], 
                 alpha=None, beta=None, nhead=2, d_model=100, dropout=0.1):
        super(Net, self).__init__()
        """Parameter initialization"""
        conv = gnn.GCNConv  
        self.feature = None
        self.alpha = alpha
        self.beta = beta
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.latent_dim = latent_dim
        self.dropout = dropout        
                      
        """Line Graph Encoder"""
        self.conv2 = nn.ModuleList()
        self.conv2.append(conv(input_dim2, latent_dim[0], cached=False))
        for i in range(1, len(latent_dim)):
            self.conv2.append(conv(latent_dim[i-1], latent_dim[i], cached=False))

        """Expression Encoder"""
        self.exp_encoder = Expression_Encoder(input_dim=input_dim1, nhead=nhead, d_model=d_model, dropout=0.2)

        """MLP layer"""
        latent_dim = sum(latent_dim)
        self.linear1 = nn.Linear(latent_dim, hidden_size)
        self.linear2 = nn.Linear(d_model+hidden_size, 1)
        self.logits1 = nn.Linear(d_model, 1)
        self.logits2 = nn.Linear(hidden_size, 1)


    def forward(self, data):
        """Data initialization"""
        data.to(torch.device("cuda"))
        # x: num_node×feature_dim    edge_index: 2×num_edges    batch: num_node×1   y: num_graph×1
        x, edge_index, batch, y = data.x, data.edge_index, data.batch, data.y
        layer2 = x
        cat_layers2 = []

        # Expression Encoder
        pair_feature = data.pair_feature
        _, hidden1 = self.exp_encoder(pair_feature)
    

        """Line graph learning graph structure information extraction"""
        lv = 0
        while lv < len(self.latent_dim):
            # Extract expression embedding
            layer2 = self.conv2[lv](layer2, edge_index)
            layer2 = torch.tanh(layer2)
            layer2 = F.dropout(layer2, p=0.25, training=self.training)
            cat_layers2.append(layer2)
            lv += 1
        # Get node embedding
        layer2 = torch.cat(cat_layers2, 1)
        

        """Extract graph embedding"""
        batch_idx = torch.unique(batch)
        idx = []
        for i in batch_idx:
            idx.append((batch==i).nonzero()[0].cpu().numpy()[0])
        layer2 = layer2[idx, :]
        hidden2 = self.linear1(layer2)

        """Fully connected layer concatenation"""
        hidden = torch.cat((hidden1, hidden2), dim=1)
        hidden = F.relu(hidden)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)

        """Calculate losses"""
        hidden1 = F.relu(hidden1)
        logits1 = self.logits1(hidden1)
        logits1 = torch.sigmoid(logits1)
        logits1 = logits1.squeeze(1)
        hidden2 = F.relu(hidden2)
        logits2 = self.logits2(hidden2)
        logits2 = torch.sigmoid(logits2)
        logits2 = logits2.squeeze(1)
        # logits2 = F.log_softmax(logits2, dim=1)

        logits = self.linear2(hidden)
        logits = torch.sigmoid(logits)
        logits = logits.squeeze(1)
        # logits = F.log_softmax(logits, dim=1)          

        alpha = self.alpha
        beta = self.beta
        if y is not None:
            y = Variable(y)
            # loss_2 = F.nll_loss(logits, y)
            loss_1 = F.binary_cross_entropy(logits1.double(), y.double())
            loss_2 = F.binary_cross_entropy(logits2.double(), y.double())
            loss_3 = F.binary_cross_entropy(logits.double(), y.double())
            loss = alpha*loss_1 + beta*loss_2 + loss_3 

            # pred = logits2.data.max(1, keepdim=True)[1]
            pred = (logits > 0.7)
            acc = pred.eq(y.data.view_as(pred)).cpu().sum().item() / float(y.size()[0])
            return logits, loss, acc, loss_1, loss_2, loss_3, hidden2
        else:
            return logits2

