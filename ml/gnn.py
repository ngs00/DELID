import torch
from torch.nn.functional import leaky_relu, normalize, elu_
from torch_geometric.nn.conv import *
from torch_geometric.nn.glob import global_add_pool


class Set2SetReadout(torch.nn.Module):
    def __init__(self, dim_in, num_timesteps):
        super(Set2SetReadout, self).__init__()
        self.num_timesteps = num_timesteps
        self.mol_gru = torch.nn.GRUCell(dim_in, dim_in)
        self.fc_out = torch.nn.Linear(dim_in, dim_in)
        self.mol_conv = GATv2Conv(dim_in, dim_in, add_self_loops=False, negative_slope=0.01)

    def reset_parameters(self):
        self.mol_gru.reset_parameters()
        self.fc_out.reset_parameters()

    def forward(self, x, batch):
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)
        out = global_add_pool(x, batch).relu_()

        for t in range(0, self.num_timesteps):
            h = elu_(self.mol_conv((x, out), edge_index))
            out = self.mol_gru(h, out).relu_()
        out = self.fc_out(out)

        return out


class GIN(torch.nn.Module):
    def __init__(self, dim_node_feat, dim_hidden, dim_out):
        super(GIN, self).__init__()
        self.dim_hidden = dim_hidden
        self.gc1 = GINConv(torch.nn.Linear(dim_node_feat, self.dim_hidden))
        self.gc2 = GINConv(torch.nn.Linear(self.dim_hidden, self.dim_hidden))
        self.readout = Set2SetReadout(self.dim_hidden, num_timesteps=4)
        self.fc = torch.nn.Linear(self.dim_hidden, dim_out)

    def forward(self, g):
        h = leaky_relu(self.gc1(g.x, g.edge_index))
        h = leaky_relu(self.gc2(h, g.edge_index))
        h = self.readout(normalize(h, p=2, dim=1), g.batch)
        out = self.fc(h)

        return out


class EGCN(torch.nn.Module):
    def __init__(self, dim_node_feat, dim_hidden, dim_out):
        super(EGCN, self).__init__()
        self.dim_hidden = dim_hidden
        self.gc1 = EGConv(dim_node_feat, self.dim_hidden)
        self.gc2 = EGConv(self.dim_hidden, self.dim_hidden)
        self.readout = Set2SetReadout(self.dim_hidden, num_timesteps=4)
        self.fc = torch.nn.Linear(self.dim_hidden, dim_out)

    def forward(self, g):
        h = leaky_relu(self.gc1(g.x, g.edge_index))
        h = leaky_relu(self.gc2(h, g.edge_index))
        h = self.readout(normalize(h, p=2, dim=1), g.batch)
        out = self.fc(h)

        return out


class GAT(torch.nn.Module):
    def __init__(self, dim_node_feat, dim_hidden, dim_out):
        super(GAT, self).__init__()
        self.dim_hidden = dim_hidden
        self.gc1 = GATv2Conv(dim_node_feat, self.dim_hidden)
        self.gc2 = GATv2Conv(self.dim_hidden, self.dim_hidden)
        self.readout = Set2SetReadout(self.dim_hidden, num_timesteps=4)
        self.fc = torch.nn.Linear(self.dim_hidden, dim_out)

    def forward(self, g):
        h = leaky_relu(self.gc1(g.x, g.edge_index))
        h = leaky_relu(self.gc2(h, g.edge_index))
        h = self.readout(normalize(h, p=2, dim=1), g.batch)
        out = self.fc(h)

        return out


class MPNN(torch.nn.Module):
    def __init__(self, dim_node_feat, dim_edge_feat, dim_hidden, dim_out):
        super(MPNN, self).__init__()
        self.dim_hidden = dim_hidden
        self.nfc = torch.nn.Linear(dim_node_feat, self.dim_hidden)
        self.efc1 = torch.nn.Sequential(torch.nn.Linear(dim_edge_feat, self.dim_hidden),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(self.dim_hidden, self.dim_hidden * self.dim_hidden))
        self.gc1 = NNConv(self.dim_hidden, self.dim_hidden, self.efc1)
        self.efc2 = torch.nn.Sequential(torch.nn.Linear(dim_edge_feat, self.dim_hidden),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(self.dim_hidden, self.dim_hidden * self.dim_hidden))
        self.gc2 = NNConv(self.dim_hidden, self.dim_hidden, self.efc2)
        self.readout = Set2SetReadout(self.dim_hidden, num_timesteps=4)
        self.fc = torch.nn.Linear(self.dim_hidden, dim_out)

    def forward(self, g):
        h = leaky_relu(self.nfc(g.x))
        h = leaky_relu(self.gc1(h, g.edge_index, g.edge_attr))
        h = leaky_relu(self.gc2(h, g.edge_index, g.edge_attr))
        h = self.readout(normalize(h, p=2, dim=1), g.batch)
        out = self.fc(h)

        return out


class CGCNN(torch.nn.Module):
    def __init__(self, dim_node_feat, dim_edge_feat, dim_hidden, dim_out):
        super(CGCNN, self).__init__()
        self.dim_hidden = dim_hidden
        self.nfc = torch.nn.Linear(dim_node_feat, self.dim_hidden)
        self.gc1 = CGConv(self.dim_hidden, dim_edge_feat)
        self.gc2 = CGConv(self.dim_hidden, dim_edge_feat)
        self.readout = Set2SetReadout(self.dim_hidden, num_timesteps=4)
        self.fc = torch.nn.Linear(self.dim_hidden, dim_out)

    def forward(self, g):
        h = leaky_relu(self.nfc(g.x))
        h = leaky_relu(self.gc1(h, g.edge_index, g.edge_attr))
        h = leaky_relu(self.gc2(h, g.edge_index, g.edge_attr))
        h = self.readout(normalize(h, p=2, dim=1), g.batch)
        out = self.fc(h)

        return out


class UniMP(torch.nn.Module):
    def __init__(self, dim_node_feat, dim_edge_feat, dim_hidden, dim_out):
        super(UniMP, self).__init__()
        self.dim_hidden = dim_hidden
        self.gc1 = TransformerConv(dim_node_feat, self.dim_hidden, edge_dim=dim_edge_feat)
        self.gc2 = TransformerConv(self.dim_hidden, self.dim_hidden, edge_dim=dim_edge_feat)
        self.readout = Set2SetReadout(self.dim_hidden, num_timesteps=4)
        self.fc = torch.nn.Linear(self.dim_hidden, dim_out)

    def forward(self, g):
        h = leaky_relu(self.gc1(g.x, g.edge_index, g.edge_attr))
        h = leaky_relu(self.gc2(h, g.edge_index, g.edge_attr))
        h = self.readout(normalize(h, p=2, dim=1), g.batch)
        out = self.fc(h)

        return out
