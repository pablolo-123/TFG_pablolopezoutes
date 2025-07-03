import os
import numpy as np
from math import sqrt

import pandas as pd
from rdkit import Chem
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch

import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, AllChem
import networkx as nx
from utils import *
class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', path='',
                 transform=None,
                 pre_transform=None):
        self.path = path
        #root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'

        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(root)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['process.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def one_of_k_encoding(self, x, allowable_set):
        if x not in allowable_set:
            raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
        return list(map(lambda s: x == s, allowable_set))

    def one_of_k_encoding_unk(self, x, allowable_set):
        """Maps inputs not in the allowable set to the last element."""
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))

    def atom_features(self, atom):
        return np.array(self.one_of_k_encoding_unk(atom.GetSymbol(),
                                              ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
                                               'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co',
                                               'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn',
                                               'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                        self.one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        self.one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        self.one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        [atom.GetIsAromatic()])


    def bond_features(self, bond):
        bond_type = self.one_of_k_encoding_unk(bond.GetBondType(), [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
            'Unknown'
        ])

        stereo = self.one_of_k_encoding_unk(bond.GetStereo(), [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE
        ])

        return np.array(bond_type + stereo + [
            bond.GetIsConjugated(),
            bond.IsInRing()
        ])


    def smile_to_graph(self, smile):
        mol = Chem.MolFromSmiles(smile)
        c_size = mol.GetNumAtoms()

        features = []
        for atom in mol.GetAtoms():
            feature = self.atom_features(atom)
            features.append(feature / sum(feature))

        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            e_feat = self.bond_features(bond)
            # Añadir en ambas direcciones (grafo dirigido)
            edge_index += [[i, j], [j, i]]
            edge_attr += [e_feat, e_feat]

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        return c_size, features, edge_index, edge_attr

    def process(self, root):
        print(self.path)
        df = pd.read_csv(root + 'raw/' + self.path)
        smile_graph = {}
        data_list = []
        data_len = len(df)
        fingerprint_vectors=[]
        for i in range(data_len):
            print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles = df.loc[i,'SMILES']
            # print(smiles)
            labels_multi = df.loc[i,'label_multi']
            labels_bin = df.loc[i, 'label_binary']
            # convert SMILES to molecular representation using rdkit
            g = self.smile_to_graph(smiles)

            # #摩根指纹
            # mol=Chem.MolFromSmiles(smiles)
            # fingerprint=AllChem.GetMACCSKeysFingerprint(mol)
            # # fingerprint=AllChem.GetMorganFingerprintAsBitVect(mol,radius=2,nBits=2048)
            # fingerprint_array=np.array(fingerprint)


            # print(g)分子图
            smile_graph[smiles] = g

            c_size, features, edge_index, edge_attr = smile_graph[smiles]
            
            #edge_tensor = torch.LongTensor(edge_index)
            
            # Manejar casos vacíos y asegurar el formato [2, num_aristas]
            if len(edge_index) == 0:
                edge_tensor = torch.empty((2, 0), dtype=torch.long)
            else:
                # Convertir lista de aristas a tensor y transponer
                edge_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            
            """
            if edge_tensor.dim() == 1:
                # Si es una lista plana o está vacío, remodelar a (num_edges, 2)
                edge_tensor = edge_tensor.view(-1, 2)
                edge_index = edge_tensor.transpose(1, 0)
                print(edge_index)
            """
            
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(
                x=torch.Tensor(features),
                finger=smiles,
                edge_index=edge_index,
                edge_attr=edge_attr,  # ¡aquí se añade!
                y_bin=torch.LongTensor([labels_bin]),
                y_multi=torch.LongTensor([labels_multi])
            )
            # GCNData.finger=torch.LongTensor([fingerprint_vectors])
            # print("GCNData",GCNData)
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])
