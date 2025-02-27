import argparse
import sys
import os
import os.path as osp
import csv

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU

from torch_geometric.nn import CGConv, GlobalAttention, NNConv, Set2Set


# Cell 2: Import statements
import argparse
import sys
import os
import os.path as osp
import csv
import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU
from torch_geometric.data import Data, DataLoader
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.nn import CGConv, GlobalAttention, NNConv, Set2Set


class PolymerNet(torch.nn.Module):
    """CGConv + Global attention pooling."""

    def __init__(self, node_in_len, edge_in_len, fea_len, n_layers, n_h):
        super(PolymerNet, self).__init__()
        self.node_embed = Linear(node_in_len, fea_len)
        self.edge_embed = Linear(edge_in_len, fea_len)
        self.cgconvs = nn.ModuleList([
            CGConv(fea_len, fea_len, aggr='mean', batch_norm=True)
            for _ in range(n_layers)])

        self.pool = GlobalAttention(
            gate_nn=Sequential(Linear(fea_len, fea_len), Linear(fea_len, 1)),
            nn=Sequential(Linear(fea_len, fea_len), Linear(fea_len, fea_len)))
        self.hs = nn.ModuleList(
            [Linear(fea_len, fea_len) for _ in range(n_h - 1)])
        self.out = Linear(fea_len, 1)

    def forward(self, data):
        out = F.leaky_relu(self.node_embed(data.x))
        edge_attr = F.leaky_relu(self.edge_embed(data.edge_attr))

        for cgconv in self.cgconvs:
            out = cgconv(out, data.edge_index, edge_attr)

        out = self.pool(out, data.batch)

        for hidden in self.hs:
            out = F.leaky_relu(hidden(out))
        out = self.out(out)
        return torch.squeeze(out, dim=-1)
    

# Cell 3: Configurations
PROP_CONFIGS = {
    'conductivity': {'mean': -4.262819, 'std': 0.222358, 'log10': True},
    'li_diffusivity': {'mean': -7.81389, 'std': 0.205920, 'log10': True},
    'poly_diffusivity': {'mean': -7.841585, 'std': 0.256285, 'log10': True},
    'tfsi_diffusivity': {'mean': -7.60879, 'std': 0.217374, 'log10': True},
    'transference_number': {'mean': 0.0623139, 'std': 0.281334, 'log10': False},
}

# Feature mappings
x_map = {
    'atomic_num': list(range(0, 119)),
    'degree': list(range(0, 11)),
    'formal_charge': list(range(-5, 7)),
    'num_hs': list(range(0, 9)),
    'hybridization': [
        'UNSPECIFIED', 'S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map = {
    'bond_type': [
        'misc', 'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC',
    ],
    'stereo': [
        'STEREONONE', 'STEREOZ', 'STEREOE', 'STEREOCIS', 'STEREOTRANS', 'STEREOANY',
    ],
    'is_conjugated': [False, True],
}

# Cell 4: Helper functions
def onehot(feature_list, cur_feature):
    assert cur_feature in feature_list
    vector = [0] * len(feature_list)
    index = feature_list.index(cur_feature)
    vector[index] = 1
    return vector

def process_smiles(smiles, form_ring, has_H):
    mol = Chem.MolFromSmiles(smiles)
    if has_H:
        mol = Chem.AddHs(mol)
    if form_ring:
        rxn = AllChem.ReactionFromSmarts('([Cu][*:1].[*:2][Au])>>[*:1]-[*:2]')
        results = rxn.RunReactants([mol])
        if not (len(results) == 1 and len(results[0]) == 1):
            rxn = AllChem.ReactionFromSmarts('([Cu]=[*:1].[*:2]=[Au])>>[*:1]=[*:2]')
            results = rxn.RunReactants([mol])
        assert len(results) == 1 and len(results[0]) == 1, smiles
        mol = results[0][0]
    Chem.SanitizeMol(mol)
    return mol


# Cell 5: Dataset Class
class PolymerDataset(Dataset):
    def __init__(self, smiles_list, log10=True, form_ring=True, has_H=True):
        self.raw_data = []
        for smiles in smiles_list:
            self.raw_data.append(['test', smiles, 1.])
        self.log10 = log10
        self.form_ring = form_ring
        self.has_H = has_H

    def __len__(self):
        return len(self.raw_data)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx):
        poly_id, smiles, target = self.raw_data[idx]
        mol = process_smiles(smiles, form_ring=self.form_ring, has_H=self.has_H)
        target = float(target)
        if self.log10:
            target = np.log10(target)
        target = torch.tensor(target).float()

        xs = []
        for atom in mol.GetAtoms():
            x = []
            x += onehot(x_map['atomic_num'], atom.GetAtomicNum())
            x += onehot(x_map['degree'], atom.GetTotalDegree())
            x += onehot(x_map['formal_charge'], atom.GetFormalCharge())
            x += onehot(x_map['num_hs'], atom.GetTotalNumHs())
            x += onehot(x_map['hybridization'], str(atom.GetHybridization()))
            x += onehot(x_map['is_aromatic'], atom.GetIsAromatic())
            x += onehot(x_map['is_in_ring'], atom.IsInRing())
            xs.append(x)

        x = torch.tensor(xs).to(torch.float)

        edge_indices, edge_attrs = [], []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            e = []
            e += onehot(e_map['bond_type'], str(bond.GetBondType()))
            e += onehot(e_map['stereo'], str(bond.GetStereo()))
            e += onehot(e_map['is_conjugated'], bond.GetIsConjugated())

            edge_indices += [[i, j], [j, i]]
            edge_attrs += [e, e]

        edge_index = torch.tensor(edge_indices)
        edge_index = edge_index.t().to(torch.long).view(2, -1)
        edge_attr = torch.tensor(edge_attrs).to(torch.float)

        # Sort indices.
        if edge_index.numel() > 0:
            perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
            edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=target,
                    smiles=smiles, poly_id=poly_id)
        return data
    

# Cell 6: Model Class
class PolymerNet(torch.nn.Module):
    def __init__(self, node_in_len, edge_in_len, fea_len, n_layers, n_h):
        super(PolymerNet, self).__init__()
        self.node_embed = Linear(node_in_len, fea_len)
        self.edge_embed = Linear(edge_in_len, fea_len)
        self.cgconvs = nn.ModuleList([
            CGConv(fea_len, fea_len, aggr='mean', batch_norm=True)
            for _ in range(n_layers)])

        self.pool = GlobalAttention(
            gate_nn=Sequential(Linear(fea_len, fea_len), Linear(fea_len, 1)),
            nn=Sequential(Linear(fea_len, fea_len), Linear(fea_len, fea_len)))
        self.hs = nn.ModuleList(
            [Linear(fea_len, fea_len) for _ in range(n_h - 1)])
        self.out = Linear(fea_len, 1)

    def forward(self, data):
        out = F.leaky_relu(self.node_embed(data.x))
        edge_attr = F.leaky_relu(self.edge_embed(data.edge_attr))

        for cgconv in self.cgconvs:
            out = cgconv(out, data.edge_index, edge_attr)

        out = self.pool(out, data.batch)

        for hidden in self.hs:
            out = F.leaky_relu(hidden(out))
        out = self.out(out)
        return torch.squeeze(out, dim=-1)

# Cell 7: Prediction Function
def predict(smiles_list, property,base_path):
    has_H, form_ring = False, True
    log10 = PROP_CONFIGS[property]['log10']
    mean, std = PROP_CONFIGS[property]['mean'], PROP_CONFIGS[property]['std']
    fea_len, n_layers, n_h = 16, 4, 2

    pred_dataset = PolymerDataset(
        smiles_list, log10=log10, form_ring=form_ring, has_H=has_H)
    pred_loader = DataLoader(
        pred_dataset, batch_size=128, shuffle=False)

    data_example = pred_dataset[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PolymerNet(
        data_example.num_features, data_example.num_edge_features,
        fea_len, n_layers, n_h).to(device)

    # You'll need to upload your pre-trained model file to Colab
    # and modify this path accordingly
    path_to_pre_trained_model = os.path.join(base_path,
        'pre_trained_gnns/{}.pth'.format(property))
    model.load_state_dict(torch.load(path_to_pre_trained_model,
                                    map_location='cpu'))

    model.eval()
    poly_ids = []
    preds = []
    targets = []
    smiles = []
    for data in pred_loader:
        data = data.to(device)
        pred = model(data)
        # De-normalize prediction
        pred = pred * std + mean
        preds.append(pred.cpu().detach().numpy())
        targets.append(data.y.cpu().detach().numpy())
        poly_ids += data.poly_id
        smiles += data.smiles
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    if log10:
        preds = 10**preds
    return preds
