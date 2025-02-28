import torch
from torch.utils.data import DataLoader
from torch.nn import PReLU
from torch.nn.functional import normalize, softplus, sigmoid
from torch_geometric.nn.conv import MessagePassing, NNConv
from torch_scatter import scatter_mean
from ml.gnn import Set2SetReadout
from util.data import collate


def num_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def _log(x):
    return torch.log(x + 1e-10)


def sample_gaussian_var(x, mean_net, std_net):
    return GaussianSample(mean_net(x), softplus(std_net(x)))


def sample_ber_var(x, p_net):
    return BerSample(sigmoid(p_net(x)))


def kl_div_ber(p1, p2):
    return p1 * _log(p1 / (p2 + 1e-10)) + (1 - p1) * _log((1 - p1) / (1 - p2 + 1e-10))


def kl_div_gaussian(mean1, std1, mean2, std2):
    loss = 2 * _log(std2 / (std1 + 1e-10)) + (std1 / (std2 + 1e-10))**2
    loss += ((mean1 - mean2) / (std2 + 1e-10))**2 - 1

    return 0.5 * loss


class GaussianSample:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.x = self.mean + torch.randn_like(self.std) * self.std**2


class BerSample:
    def __init__(self, prob):
        self.prob = prob
        eps = torch.rand_like(self.prob)
        self.x = sigmoid(_log(eps) - _log(1 - eps) + _log(self.prob) - _log(1 - self.prob))


class AtomEmbeddingBlock(torch.nn.Module):
    def __init__(self, dim_edge_feat, dim_hidden):
        super(AtomEmbeddingBlock, self).__init__()
        self.efc = torch.nn.Sequential(torch.nn.Linear(dim_edge_feat, dim_hidden),
                                       torch.nn.PReLU(),
                                       torch.nn.Linear(dim_hidden, dim_hidden * dim_hidden))
        self.gc = NNConv(dim_hidden, dim_hidden, self.efc)
        self.act_gc = PReLU()

    def forward(self, x, edge_index, edge_attr):
        return self.act_gc(self.gc(x, edge_index, edge_attr))


class DELID(MessagePassing):
    def __init__(self, dim_node_feat, dim_edge_feat, dim_latent, dim_hidden, gnn_frag):
        super(DELID, self).__init__()
        self.dim_hidden = dim_hidden
        self.dim_latent = dim_latent
        self.gnn_frag = gnn_frag

        # Atom embedding layer.
        self.fc_node = torch.nn.Linear(dim_node_feat, self.dim_hidden)
        self.act_fc_node = PReLU()
        self.gc1 = AtomEmbeddingBlock(dim_edge_feat, dim_hidden)
        self.gc2 = AtomEmbeddingBlock(dim_edge_feat, dim_hidden)
        self.act_gc1 = PReLU()
        self.act_gc2 = PReLU()

        # Edge embedding layer.
        self.fc_edge = torch.nn.Linear(2 * self.dim_hidden, self.dim_hidden)
        self.act_fc_edge = PReLU()

        # Molecule embedding layer.
        self.readout = Set2SetReadout(self.dim_hidden, num_timesteps=4)
        self.fc_mol_emb = torch.nn.Linear(self.dim_hidden, self.dim_hidden)

        # prediction layer.
        self.fc_out = torch.nn.Linear(self.dim_hidden, 1)

        # diffusion layers.
        self.fc_ber_p_p = torch.nn.Linear(self.dim_latent, 1)
        self.fc_ber_p_q = torch.nn.Linear(self.dim_latent, 1)
        self.fc_mean_z_q = torch.nn.Linear(self.dim_hidden, self.dim_latent)
        self.fc_std_z_q = torch.nn.Linear(self.dim_hidden, self.dim_latent)
        self.fc_mean_z_p = torch.nn.Linear(1 + self.dim_hidden, self.dim_latent)
        self.fc_std_z_p = torch.nn.Linear(1 + self.dim_hidden, self.dim_latent)
        self.fc_prior = torch.nn.Linear(self.dim_hidden, 1)

        # Fragmented information embedding layer.
        self.fc_mean_zfrag = torch.nn.Linear(2 * self.dim_hidden + self.dim_latent, self.dim_latent)
        self.fc_std_zfrag = torch.nn.Linear(2 * self.dim_hidden + self.dim_latent, self.dim_latent)
        self.fc_mean_yfrag = torch.nn.Linear(self.dim_hidden + self.dim_latent + 1, 1)
        self.fc_std_yfrag = torch.nn.Linear(self.dim_hidden + self.dim_latent + 1, 1)

        self.fc_ber_p_joint = torch.nn.Linear(self.dim_hidden + self.dim_latent + 1, 1)
        self.fc_mean_z_joint = torch.nn.Linear(2 * self.dim_hidden + self.dim_latent, self.dim_latent)
        self.fc_std_z_joint = torch.nn.Linear(2 * self.dim_hidden + self.dim_latent, self.dim_latent)

        # Initialize model parameters.
        self.reset_parameters()

    def reset_parameters(self):
        self.fc_node.reset_parameters()
        self.fc_edge.reset_parameters()
        self.readout.reset_parameters()
        self.fc_mol_emb.reset_parameters()

        self.fc_ber_p_p.reset_parameters()
        self.fc_ber_p_q.reset_parameters()
        self.fc_mean_z_q.reset_parameters()
        self.fc_std_z_q.reset_parameters()
        self.fc_mean_z_p.reset_parameters()
        self.fc_std_z_p.reset_parameters()
        self.fc_prior.reset_parameters()

        self.fc_mean_zfrag.reset_parameters()
        self.fc_std_zfrag.reset_parameters()
        self.fc_mean_yfrag.reset_parameters()
        self.fc_std_yfrag.reset_parameters()
        self.fc_ber_p_joint.reset_parameters()
        self.fc_mean_z_joint.reset_parameters()
        self.fc_std_z_joint.reset_parameters()

    def calc_recon_loss(self, z_q, batch_edge):
        p_q = sigmoid(self.fc_ber_p_q(z_q.x))
        p_p = sigmoid(self.fc_ber_p_p(z_q.x))
        loss_recon = scatter_mean(p_q * _log(p_p) + (1 - p_q) * _log(1 - p_p), batch_edge, dim=0)
        loss_recon = -torch.sum(loss_recon)

        return loss_recon

    def calc_prior_match_loss(self, z_q, z_gfrag, batch_edge):
        p_q = sigmoid(self.fc_ber_p_q(z_q.x))
        p_prior = sigmoid(self.fc_prior(z_gfrag))[batch_edge]
        loss = p_q * _log(p_q / (p_prior + 1e-10)) + (1 - p_q) * _log((1 - p_q) / (1 - p_prior + 1e-10))
        loss = torch.sum(scatter_mean(loss, batch_edge, dim=0))

        return loss

    def calc_trans_match_loss(self, z_q, z_p, batch_edge):
        loss = torch.sum(kl_div_gaussian(z_q.mean, z_q.std, z_p.mean, z_p.std), dim=1, keepdim=True)
        loss = torch.sum(scatter_mean(loss, batch_edge, dim=0))

        return loss

    def calc_frag_diff_loss(self, z_p, x_joint, z_joint, z_frag, y_frag, batch_edge):
        p_trans = sigmoid(self.fc_ber_p_p(z_p.x))
        p_joint = sigmoid(self.fc_ber_p_joint(z_joint))[batch_edge]
        z_p_joint = sample_gaussian_var(x_joint, self.fc_mean_z_joint, self.fc_std_z_joint)
        z_p_joint_mean = z_p_joint.mean[batch_edge]
        z_p_joint_std = z_p_joint.std[batch_edge]

        loss_kl1 = p_trans * _log(p_trans / p_joint) + (1 - p_trans) * _log((1 - p_trans) / (1 - p_joint + 1e-10))
        loss_kl1 = scatter_mean(loss_kl1, batch_edge, dim=0)
        loss_kl2 = kl_div_gaussian(z_p.mean, z_p.std, z_p_joint_mean, z_p_joint_std)
        loss_kl2 = torch.sum(scatter_mean(loss_kl2, batch_edge, dim=0), dim=1, keepdim=True)

        loss_exp1 = _log(y_frag.std) + 0.5 * ((y_frag.x - y_frag.mean) / (y_frag.std + 1e-10))**2
        loss_exp2 = 0.5 * (z_frag.x - z_frag.mean)**2
        loss_exp2 = torch.sum(loss_exp2, dim=1, keepdim=True)

        loss = torch.sum(loss_kl1 + loss_kl2 + loss_exp1 + loss_exp2)

        return torch.sum(loss)

    def emb_atom(self, g):
        hx = self.act_fc_node(self.fc_node(g.x))
        hx = normalize(self.act_gc1(self.gc1(hx, g.edge_index, g.edge_attr)), p=2, dim=1)
        hx = normalize(self.act_gc2(self.gc2(hx, g.edge_index, g.edge_attr)), p=2, dim=1)
        he = torch.cat([hx[g.edge_index[0]], hx[g.edge_index[1]]], dim=1)
        he = normalize(self.act_fc_edge(self.fc_edge(he)), p=2, dim=1)

        return hx, he

    def emb(self, g, g_frag, batch_edge, edge_label):
        hx, he = self.emb_atom(g)
        z_mol = self.fc_mol_emb(self.readout(he, batch_edge))
        z_p = sample_gaussian_var(torch.cat([edge_label, he], dim=1), self.fc_mean_z_p, self.fc_std_z_p)

        zgf = self.gnn_frag(g_frag)
        _z_p = scatter_mean(z_p.x, batch_edge, dim=0)
        x_joint = torch.cat([zgf, z_mol, _z_p], dim=1)
        z_frag = sample_gaussian_var(x_joint, self.fc_mean_zfrag, self.fc_std_zfrag)

        x_0 = sample_ber_var(z_p.x, self.fc_ber_p_p)
        _x_0 = scatter_mean(x_0.x, batch_edge, dim=0)
        z_joint = torch.cat([z_frag.x, _x_0], dim=1)

        return z_joint

    def forward(self, g, g_frag, batch_edge, edge_label):
        # Atom embedding from the input graph.
        # hx: n_atoms * dim_hidden
        # he: n_edges * dim_hidden
        hx, he = self.emb_atom(g)

        # Molecular embedding from the input graph.
        # z_mol: batch_size * dim_hidden
        z_mol = self.fc_mol_emb(self.readout(he, batch_edge))

        # Adjacency matrix embedding from the input edge embeddings.
        # z_q: n_edges * dim_latent
        z_q = sample_gaussian_var(he, self.fc_mean_z_q, self.fc_std_z_q)

        # Adjacency matrix embedding from the decomposed edges.
        # z_p: n_edges * dim_latent
        z_p = sample_gaussian_var(torch.cat([edge_label, he], dim=1), self.fc_mean_z_p, self.fc_std_z_p)

        # Electron-level information embedding from the fragmented electron-level information.
        zgf = self.gnn_frag(g_frag)
        _z_p = scatter_mean(z_p.x, batch_edge, dim=0)
        x_joint = torch.cat([zgf, z_mol, _z_p], dim=1)
        z_frag = sample_gaussian_var(x_joint, self.fc_mean_zfrag, self.fc_std_zfrag)

        # Prediction of the target variable.
        # y_frag: batch_size * 1
        x_0 = sample_ber_var(z_p.x, self.fc_ber_p_p)
        _x_0 = scatter_mean(x_0.x, batch_edge, dim=0)
        z_joint = torch.cat([z_frag.x, z_mol, _x_0], dim=1)
        y_frag = sample_gaussian_var(z_joint, self.fc_mean_yfrag, self.fc_std_yfrag)
        y_g = self.fc_out(z_mol)
        out_y = y_g + y_frag.x

        # Decomposition diffusion losses.
        loss_recon = self.calc_recon_loss(z_q, batch_edge)
        loss_prior_match = self.calc_prior_match_loss(z_q, zgf, batch_edge)
        loss_trans_match = self.calc_trans_match_loss(z_q, z_p, batch_edge)

        # Information diffusion loss.
        loss_info_diffu = self.calc_frag_diff_loss(z_p, x_joint, z_joint, z_frag, y_frag, batch_edge)

        return out_y, y_g + y_frag.mean, y_frag.std, loss_recon, loss_prior_match, loss_trans_match, loss_info_diffu

    def fit(self, data_loader, optimizer):
        train_loss = 0

        self.train()
        for batch, batch_frag, batch_edge, labels, targets in data_loader:
            batch = batch.cuda()
            batch_frag = batch_frag.cuda()
            batch_edge = batch_edge.cuda()
            labels = labels.cuda()
            targets = targets.cuda()

            preds, y_mean, y_std, loss_recon, loss_prior_match, loss_trans_match, loss_info_diffu = self(batch,
                                                                                                         batch_frag,
                                                                                                         batch_edge,
                                                                                                         labels)
            loss_pred = torch.sum(_log(y_std) + 0.5 * ((targets - y_mean) / (y_std + 1e-10)) ** 2)
            loss = loss_pred + loss_recon + loss_prior_match + loss_trans_match + loss_info_diffu

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        return train_loss / len(data_loader)

    def predict(self, dataset_test, y_mean=None, y_std=None):
        data_loader = DataLoader(dataset_test, batch_size=64, collate_fn=collate)
        list_preds = list()

        self.eval()
        with torch.no_grad():
            for batch, batch_frag, batch_edge, labels, _ in data_loader:
                batch = batch.cuda()
                batch_frag = batch_frag.cuda()
                batch_edge = batch_edge.cuda()
                labels = labels.cuda()
                preds, _, _, _, _, _, _ = self(batch, batch_frag, batch_edge, labels)

                list_preds.append(preds)

        preds = torch.vstack(list_preds).cpu()

        if y_mean is not None and y_std is not None:
            preds = y_std * preds + y_mean

        return preds.flatten()

    def get_embs(self, dataset_test):
        data_loader = DataLoader(dataset_test, batch_size=64, collate_fn=collate)
        list_embs = list()

        self.eval()
        with torch.no_grad():
            for batch, batch_frag, batch_edge, labels, _ in data_loader:
                batch = batch.cuda()
                batch_frag = batch_frag.cuda()
                batch_edge = batch_edge.cuda()
                labels = labels.cuda()
                embs = self.emb(batch, batch_frag, batch_edge, labels)

                list_embs.append(embs)

        return torch.vstack(list_embs).cpu()
