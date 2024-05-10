import torch as th
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv



class ActorNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256, num_layers=2, gnn_type='gcn'):
        super(ActorNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gnn_type = gnn_type

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            if gnn_type == 'gcn':
                self.gnn_layers.append(GCNConv(hidden_size, hidden_size))
            elif gnn_type == 'gat':
                self.gnn_layers.append(GATConv(hidden_size, hidden_size // 8, heads=8))

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.relu(self.fc1(x))

        for layer in self.gnn_layers:
            if self.gnn_type == 'gcn':
                x = self.relu(layer(x, edge_index))
            elif self.gnn_type == 'gat':
                x = self.relu(layer(x, edge_index))
                x = F.dropout(x, p=0.6, training=self.training)

        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x
