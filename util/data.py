import pandas
import numpy
import torch
import re
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs import TanimotoSimilarity
from EFGs import mol2frag
from torch_geometric.data.batch import Batch
from torch_geometric.data import Data
from tqdm import tqdm
from itertools import chain
from sklearn.preprocessing import scale
from util.chem import get_mol_graph


class MolDecompData:
    def __init__(self, mg, mol, frags, frag_idx, adj_mat, decomp_adj_mat, frag_mg, target):
        self.mg = mg
        self.mol = mol
        self.frags = frags
        self.frag_idx = frag_idx
        self.adj_mat = torch.tensor(adj_mat, dtype=torch.long)
        self.decomp_adj_mat = torch.tensor(decomp_adj_mat, dtype=torch.long)
        self.frag_mg = frag_mg
        self.target = torch.tensor(target, dtype=torch.float)
        self.labels = list()

        if self.mg.edge_index.shape[1] == 0:
            self.labels = None
        else:
            edge_idx = self.mg.edge_index
            for i in range(0, edge_idx.shape[1]):
                self.labels.append(self.decomp_adj_mat[edge_idx[0][i]][edge_idx[1][i]])
            self.labels = torch.vstack(self.labels).float()


class MolDecompDataset:
    def __init__(self, data, normalize_y):
        self.data = data
        self.normalize_y = normalize_y
        self.y = torch.tensor([d.target for d in data], dtype=torch.float)
        self.y_mean = torch.mean(self.y)
        self.y_std = torch.std(self.y)

        if self.normalize_y:
            self.y = (self.y - self.y_mean) / self.y_std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].mg, self.data[idx].frag_mg, self.data[idx].labels, self.y[idx]

    def get_original_y(self):
        return torch.tensor([d.target for d in self.data], dtype=torch.float)

    def get_k_folds(self, n_folds, random_seed=None):
        if random_seed is not None:
            numpy.random.seed(random_seed)

        k_folds = list()
        idx_rand = numpy.array_split(numpy.random.permutation(len(self.data)), n_folds)

        for i in range(0, n_folds):
            idx_train = list(chain.from_iterable(idx_rand[:i] + idx_rand[i + 1:]))
            idx_test = idx_rand[i]
            dataset_train = MolDecompDataset([self.data[idx] for idx in idx_train], self.normalize_y)
            dataset_test = MolDecompDataset([self.data[idx] for idx in idx_test], self.normalize_y)
            k_folds.append([dataset_train, dataset_test])

        return k_folds


def __decompose_mol(mol):
    decomp_results = mol2frag(mol, returnidx=True)

    for i in range(0, len(decomp_results[1])):
        decomp_results[1][i] = re.sub('0[0-9]0', '', decomp_results[1][i])

    frags = decomp_results[0] + decomp_results[1]
    frag_idx = [list(idx) for idx in decomp_results[2] + decomp_results[3]]

    return frags, frag_idx


def decompose(smiles, dataset_calc, elem_attrs, fpgen, target):
    mol = Chem.MolFromSmiles(smiles)
    frags, frag_idx = __decompose_mol(mol)
    adj_mat = Chem.GetAdjacencyMatrix(mol)
    decomp_adj_mat = numpy.zeros(adj_mat.shape, dtype=numpy.int32)
    mg = get_mol_graph(smiles, elem_attrs)

    if mg is None:
        return None

    frag_feats = get_frag_feats(frags, dataset_calc, fpgen)
    edges = get_frag_edges(frags, frag_idx, adj_mat)
    frag_mg = Data(x=frag_feats, edge_index=edges)

    for idx in frag_idx:
        for i in range(0, len(idx)):
            for j in range(0, len(idx)):
                decomp_adj_mat[idx[i], idx[j]] = adj_mat[idx[i], idx[j]]

    return MolDecompData(mg, smiles, frags, frag_idx, adj_mat, decomp_adj_mat, frag_mg, target)


def get_frag_feats(frags, dataset_calc, fpgen):
    frag_feats = list()

    for frag in frags:
        mol = Chem.MolFromSmiles(frag)

        if mol is None:
            frag_feats.append(torch.zeros((1, dataset_calc[0].calc_feats.shape[0])))
        else:
            fp = fpgen.GetSparseCountFingerprint(Chem.MolFromSmiles(frag))
            sims = list()
            for smol in dataset_calc:
                tanimoto_sim = TanimotoSimilarity(fp, smol.fp)
                sims.append(tanimoto_sim)
            frag_feats.append(dataset_calc[numpy.argmax(sims)].calc_feats.view(1, -1))

    return torch.cat(frag_feats, dim=0)


def get_frag_edges(frags, frag_idx, adj_mat):
    edges = list()

    if len(frags) == 1:
        return torch.tensor([[0], [0]], dtype=torch.long)

    for i in range(0, len(frags)):
        adj_row = adj_mat[frag_idx[i]]
        for j in range(0, len(frags)):
            if i != j:
                if numpy.sum(adj_row[:, frag_idx[j]]) > 0:
                    edges.append([i, j])

    return torch.tensor(numpy.vstack(edges), dtype=torch.long).t().contiguous()


def load_calc_dataset(path_dataset, elem_attrs, idx_smiles, idx_calc_feats):
    data = pandas.read_excel(path_dataset).values.tolist()
    fpgen = GetMorganGenerator(radius=2)
    calc_feats = list()
    dataset = list()

    for d in data:
        calc_feats.append([d[idx] for idx in idx_calc_feats])
    calc_feats = torch.tensor(scale(numpy.vstack(calc_feats)), dtype=torch.float)

    for i in tqdm(range(0, len(data))):
        mol = Chem.MolFromSmiles(data[i][idx_smiles])
        mg = get_mol_graph(data[i][idx_smiles], elem_attrs)

        if mg is not None:
            mg.calc_feats = calc_feats[i]
            mg.fp = fpgen.GetSparseCountFingerprint(mol)
            dataset.append(mg)

    return dataset


def load_dataset(path_dataset, elem_attrs, idx_smiles, idx_target, dataset_calc, normalize_y):
    data = pandas.read_excel(path_dataset).values.tolist()
    fpgen = GetMorganGenerator(radius=2)
    decomp_data = list()

    for i in tqdm(range(0, len(data))):
        d = decompose(data[i][idx_smiles], dataset_calc, elem_attrs, fpgen, data[i][idx_target])

        if d is not None and d.labels is not None:
            decomp_data.append(d)

    return MolDecompDataset(decomp_data, normalize_y=normalize_y)


def collate(batch):
    mg = list()
    mg_frag = list()
    batch_edge = list()
    labels = list()
    targets = list()

    for i in range(0, len(batch)):
        mg.append(batch[i][0])
        mg_frag.append(batch[i][1])
        batch_edge.append(torch.full((batch[i][0].edge_index.shape[1],), i))
        labels.append(batch[i][2])
        targets.append(batch[i][3])

    mg = Batch.from_data_list(mg)
    mg_frag = Batch.from_data_list(mg_frag)
    batch_edge = torch.cat(batch_edge, dim=0).long()
    labels = torch.vstack(labels)
    targets = torch.vstack(targets)

    return mg, mg_frag, batch_edge, labels, targets
