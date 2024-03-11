import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GraphConv,TransformerConv,GCNConv
from torch_geometric.seed import seed_everything

def corrcoef(input, target):
    input_ = input-input.mean(axis=0)
    target_ = target-target.mean(axis=0)
    input_ = input_/input_.norm(dim=0)
    target_ = target_/target_.norm(dim=0)
    return (input_ * target_).sum(axis=0).mean()

torch.manual_seed(12345)

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels,input_dim,num_classes,num_layers):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv_c=torch.nn.ModuleList()
        self.bn_c=torch.nn.ModuleList()
        for _ in range(int(num_layers)):
            self.conv_c.append(GCNConv(hidden_channels, hidden_channels))
            self.bn_c.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edges, batch):
        x = self.conv1(x, edges)
        x = self.bn1(x)
        x = x.relu()
        if len(self.conv_c) > 0:
            for index,conv_c_i in enumerate(self.conv_c):
                x = conv_c_i(x,edges)
                x = self.bn_c[index](x)
                x = x.relu()
        ## 2. Readout layer
        #x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x = self.lin(x)
        return x

class GTN(torch.nn.Module):
    def __init__(self, hidden_channels,input_dim,num_classes,num_layers):
        super(GTN, self).__init__()
        self.conv1 = TransformerConv(input_dim, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv_c=torch.nn.ModuleList()
        self.bn_c=torch.nn.ModuleList()
        for _ in range(int(num_layers)):
            self.conv_c.append(TransformerConv(hidden_channels, hidden_channels))
            self.bn_c.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edges, batch):
        x = self.conv1(x, edges)
        x = self.bn1(x)
        x = x.relu()
        if len(self.conv_c) > 0:
            for index,conv_c_i in enumerate(self.conv_c):
                x = conv_c_i(x,edges)
                x = self.bn_c[index](x)
                x = x.relu()
        # 2. Readout layer
        #x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x = self.lin(x)
        return x


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels,input_dim,num_classes,num_layers):
        super(GNN, self).__init__()
        self.conv1 = GraphConv(input_dim, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv_c=torch.nn.ModuleList()
        self.bn_c=torch.nn.ModuleList()
        for _ in range(int(num_layers)):
            self.conv_c.append(GraphConv(hidden_channels, hidden_channels))
            self.bn_c.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edges, batch):
        x = self.conv1(x, edges)
        x = self.bn1(x)
        x = x.relu()
        if len(self.conv_c) > 0:
            for index,conv_c_i in enumerate(self.conv_c):
                x = conv_c_i(x,edges)
                x = self.bn_c[index](x)
                x = x.relu()
        # 2. Readout layer
        #x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x = self.lin(x)
        return x

def train(model,train_loader,optimizer,corr=True):
    model.train()
    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = 0.
        for i in range(data.num_graphs): # Calculate batch-wise loss 
            loss += F.mse_loss(out[torch.where(data.batch==i)], data.y[torch.where(data.batch==i)])
            if corr:
                loss += 1 - corrcoef(out[torch.where(data.batch==i)], data.y[torch.where(data.batch==i)])
        loss = loss/data.num_graphs  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(model,loader):
    model.eval()
    mse = 0.
    pcc = 0.
    with torch.no_grad():
        for data in loader:  # Iterate in batches over the training/test dataset.
            out = model(data.x, data.edge_index, data.batch)  
            for i in range(data.num_graphs):
                mse += F.mse_loss(out[torch.where(data.batch==i)], data.y[torch.where(data.batch==i)])
                pcc += corrcoef(out[torch.where(data.batch==i)], data.y[torch.where(data.batch==i)])
    mse = mse.detach().numpy() / len(loader.dataset) # Calculate average mse over sections
    pcc = pcc.detach().numpy() / len(loader.dataset) # Calculate average pcc over sections
    return mse, pcc

def calc_loss(model,loader,corr=True):
    model.eval()
    loss = 0.
    with torch.no_grad():
        for data in loader:  # Iterate in batches over the training/test dataset.
            out = model(data.x, data.edge_index, data.batch)  
            for i in range(data.num_graphs):
                loss += F.mse_loss(out[torch.where(data.batch==i)], data.y[torch.where(data.batch==i)])
                if corr:
                    loss += 1 - corrcoef(out[torch.where(data.batch==i)], data.y[torch.where(data.batch==i)])
    loss = loss.detach().numpy() / len(loader.dataset) # Calculate average loss over sections
    return loss

# This function is mainly used to look at ST-Net mse on test set, will be removed later
def peek(loader):
    mse = 0.
    pcc = 0.
    for data in loader:  # Iterate in batches over the training/test dataset.
        for i in range(data.num_graphs):
            mse += F.mse_loss(data.y_hat[torch.where(data.batch==i)], data.y[torch.where(data.batch==i)])
            pcc += corrcoef(data.y_hat[torch.where(data.batch==i)], data.y[torch.where(data.batch==i)])
    mse = mse.detach().numpy() / len(loader.dataset) # Calculate average mse over sections
    pcc = pcc.detach().numpy() / len(loader.dataset) # Calculate average pcc over sections
    return mse, pcc

# Generate raw and predicted gene expression with model provided
def predict(model, loader):
    model.eval()
    raw = []; pred = []
    with torch.no_grad():
        for data in loader:
            raw.append(data.y)
            pred.append(model(data.x, data.edge_index, data.batch))  # Perform a single forward pass.
    return raw, pred
