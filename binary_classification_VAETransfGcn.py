#!/usr/bin/env python
# # Import
from chemprop.features import morgan_binary_features_generator


from numpy.random import seed
import sqlite3
import time
import numpy as np
import random
import pandas as pd
from pandas import DataFrame
import math
import matplotlib.pyplot as plt
from rdkit import Chem

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc, average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize

import warnings
import os
import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.nn import TransformerConv
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric.data import Data
from tqdm import tqdm
from torch_geometric.nn import global_mean_pool as gap,global_max_pool as gmp

from multiprocessing import Pool
from pubchemfp import GetPubChemFPs
from utils import *
seed = 1
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用GPU编号，可以根据需要更改



file_path = "./"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


learn_rating = 1e-3
weight_decay_rate = 1e-3
patience = 20
delta = 0
label_num_multi = 5
label_num_binary = 1
BATCH_SIZE = 128
import multiprocessing as mp

if __name__ == '__main__':
    mp.set_start_method('spawn')

# ### Early stopping


class EarlyStopping:
    def __init__(self, patience=7, verbose=True, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        # print("val_loss={}".format(val_loss))
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
        torch.save(model.state_dict(), path + "/" + "model_checkpoint.pth")
        self.val_loss_min = val_loss


# ### Ranger Optimizer

# In[10]:


# https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
from torch.optim.optimizer import Optimizer


class Ranger(Optimizer):

    def __init__(self, params, lr=1e-3,  # lr
                 alpha=0.5, k=6, N_sma_threshhold=5,  # Ranger options
                 betas=(.95, 0.999), eps=1e-5, weight_decay=0,  # Adam options
                 # Gradient centralization on or off, applied to conv layers only or conv + fc layers
                 use_gc=True, gc_conv_only=False
                 ):

        # parameter checks
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        if not lr > 0:
            raise ValueError(f'Invalid Learning Rate: {lr}')
        if not eps > 0:
            raise ValueError(f'Invalid eps: {eps}')

        # prep defaults and init torch.optim base
        defaults = dict(lr=lr, alpha=alpha, k=k, step_counter=0, betas=betas,
                        N_sma_threshhold=N_sma_threshhold, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # adjustable threshold
        self.N_sma_threshhold = N_sma_threshhold

        # look ahead params

        self.alpha = alpha
        self.k = k

        # radam buffer for state
        self.radam_buffer = [[None, None, None] for ind in range(10)]

        # gc on or off
        self.use_gc = use_gc

        # level of gradient centralization
        self.gc_gradient_threshold = 3 if gc_conv_only else 1

        print(
            f"Ranger optimizer loaded. \nGradient Centralization usage = {self.use_gc}")
        if (self.use_gc and self.gc_gradient_threshold == 1):
            print(f"GC applied to both conv and fc layers")
        elif (self.use_gc and self.gc_gradient_threshold == 3):
            print(f"GC applied to conv layers only")

    def __setstate__(self, state):
        print("set state called")
        super(Ranger, self).__setstate__(state)

    def step(self, closure=None):
        loss = None

        # Evaluate averages and grad, update param tensors
        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()

                if grad.is_sparse:
                    raise RuntimeError(
                        'Ranger optimizer does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]  # get state dict for this param

                if len(state) == 0:

                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)

                    # look ahead weight storage now in state dict
                    state['slow_buffer'] = torch.empty_like(p.data)
                    state['slow_buffer'].copy_(p.data)

                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(
                        p_data_fp32)

                # begin computations
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # GC operation for Conv layers and FC layers
                if grad.dim() > self.gc_gradient_threshold:
                    grad.add_(-grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))

                state['step'] += 1

                # compute variance mov avg
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                # compute mean moving avg
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                buffered = self.radam_buffer[int(state['step'] % 10)]

                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    if N_sma > self.N_sma_threshhold:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (
                                N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay']
                                     * group['lr'], p_data_fp32)

                # apply lr
                if N_sma > self.N_sma_threshhold:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size *
                                         group['lr'], exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)

                p.data.copy_(p_data_fp32)

                # integrated look ahead...
                # we do it at the param level instead of group level
                if state['step'] % group['k'] == 0:
                    # get access to slow param tensor
                    slow_p = state['slow_buffer']
                    # (fast weights - slow weights) * alpha
                    slow_p.add_(self.alpha, p.data - slow_p)
                    # copy interpolated weights to RAdam param tensor
                    p.data.copy_(slow_p)

        return loss


class DDIDataset(Dataset):
    def __init__(self, x, y_multi,y_binary):
        print('pppppppppppppppppppp')
        self.len = len(x)
        self.x_data=x
        self.y_multi_data=y_multi
        self.y_binary_data=y_binary
    def __getitem__(self, index):
        return self.x_data[index], self.y_multi_data[index], self.y_binary_data[index]

    def __len__(self):
        return self.len


import torch.nn as nn


from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv, GCNConv, GINConv, global_add_pool

class MultiTaskModel(nn.Module):
    def __init__(self, num_classes_multi, num_classes_binary):
        super(MultiTaskModel, self).__init__()

         # 创建GAT层
        self.conv1 = GATConv(in_channels=78, out_channels=2048, heads=8, concat=False)
        self.conv2 = GATConv(in_channels=2048, out_channels=1024, heads=8, concat=False)
        self.conv3 = GATConv(in_channels=1024, out_channels=512, heads=8, concat=False)
        self.conv4 = GCNConv(512,256)
        self.trans1 = TransformerConv(in_channels=78, out_channels=512, heads=4, concat=False)

        self.fp_1_dim = 1024
        self.fp_2_dim = 128
        self.fp_3_dim = 256
        # self.dropout_fpn = 0.2
        # self.hidden_dim = 512
        self.fp_type = 'mixed'

        if self.fp_type == 'mixed':
            self.fp_dim = 1489
        else:
            self.fp_dim = 1024

        self.fc1 = nn.Linear(self.fp_dim, self.fp_2_dim)
        self.fc2 = nn.Linear(self.fp_2_dim, self.fp_3_dim)
        self.act_func = nn.ReLU()

        self.fc_binary = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes_binary)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x, finger, edge_index, batch = data.x, data.finger, data.edge_index, data.batch

                # 使用GAT层进行特征提取
        # print('x.shape',x.shape)
        x = self.conv1(x.float(), edge_index)
        x = F.relu(x)
        x=F.dropout(x,training=self.training)
        x = self.conv2(x.float(), edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x.float(), edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv4(x.float(), edge_index)
        x_gat = gap(x, batch)  # Salida de GAT+GCN
        # 进一步处理 x，包括全局池化等

        fp_list = []
        for i, one in enumerate(finger):
            fp = []
            mol = Chem.MolFromSmiles(one)
            if self.fp_type == 'mixed':
                fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)
                fp_phaErGfp = AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1)
                fp_pubcfp = GetPubChemFPs(mol)
                fp.extend(fp_maccs)
                fp.extend(fp_phaErGfp)
                fp.extend(fp_pubcfp)
            else:
                fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                fp.extend(fp_morgan)
            fp_list.append(fp)
        fp_list = torch.Tensor(fp_list).to(device)
        fpn_out = self.fc1(fp_list)
        fpn_out = self.act_func(fpn_out)
        fpn_out = self.fc2(fpn_out)
        fpn_out = self.act_func(fpn_out)

        # Aplica TransformerConv directamente a features originales
        x_trans = self.trans1(data.x.float(), data.edge_index)
        x_trans = F.relu(x_trans)
        x_trans = gap(x_trans, batch)
        # Fusiona ambas representaciones
        x = torch.cat([x_gat, x_trans], dim=1)  # Resultado: [batch_size, 512+512=1024]
        xc = torch.cat((x, fpn_out), dim=1)  # [batch_size, 1024+256]

        # 二分类任务的预测
        y_binary = self.fc_binary(xc)
        y_binary = self.sigmoid(y_binary)

        return y_binary



class BinaryClassLoss(nn.Module):
    def __init__(self):
        super(BinaryClassLoss, self).__init__()

    def forward(self, pred, target):

        target = target.view(-1, 1)
        criterion = nn.BCELoss()
        loss = criterion(pred, target.float()) 
        return loss

# # Training

# In[ ]:


from torch.optim.lr_scheduler import CosineAnnealingLR

#没有数据增强
train_epochs_loss = []
valid_epochs_loss = []
def cross_validate_kfolds(fold):

    print(f"\n--- Fold {fold} ---")
    # rutas específicas por fold
    train_path = f'./folds/fold_{fold}/train/'
    val_path   = f'./folds/fold_{fold}/val/'
    test_path  = f'./folds/fold_{fold}/test/'

    # loaders
    train_loader = DataLoader(
        TestbedDataset(root=train_path, path='train_graph_dataset.csv'),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TestbedDataset(root=val_path, path='val_graph_dataset.csv'),
        batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        TestbedDataset(root=test_path, path='test_graph_dataset.csv'),
        batch_size=BATCH_SIZE, shuffle=False
    )

    # modelo y optimizador
    model = VAEClassifier(num_classes_binary=label_num_binary)
    optimizer = Ranger(
        model.parameters(),
        lr=learn_rating,
        weight_decay=weight_decay_rate,
        betas=(0.95, 0.999),
        eps=1e-6
    )

    # entrenamiento + evaluación
    metrics = train_fn(model, optimizer, train_loader, val_loader, test_loader,300, fold)
    metrics["fold"] = fold
    return metrics




def vae_loss_function(recon_x, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + kl_div
    return total, recon_loss.item(), kl_div.item()

class FingerprintVAE(nn.Module):
    def __init__(self, input_dim=1489, latent_dim=256):
        super(FingerprintVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(256, latent_dim)
        self.logvar_layer = nn.Linear(256, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return z, mu, logvar, recon_x

class VAEClassifier(nn.Module):
    def __init__(self, num_classes_binary=1, fingerprint_dim=1489, latent_dim=256):
        super(VAEClassifier, self).__init__()

    
        self.trans1 = TransformerConv(in_channels=78, out_channels=512, heads=4, concat=False)
        self.trans2 = TransformerConv(in_channels=512, out_channels=512, heads=4, concat=False)
        self.trans3 = TransformerConv(in_channels=512, out_channels=512, heads=4, concat=False)
        self.conv4 = GCNConv(512, 256)

        # VAE encoder
        self.vae = FingerprintVAE(input_dim=fingerprint_dim, latent_dim=latent_dim)

        # Projection for gate fusion
        self.g_proj = nn.Linear(768, 256)  # 256 (GAT+GCN) + 512 (Transformer)
        self.fp_proj = nn.Linear(latent_dim, 256)

        # Final classifier
        self.fc_binary = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes_binary)
        )
        self.sigmoid = nn.Sigmoid()
        self.fp_type = 'mixed'

    def forward(self, data):
        x, finger, edge_index, batch = data.x, data.finger, data.edge_index, data.batch

        x = self.trans1(x.float(), edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.trans2(x.float(), edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.trans3(x.float(), edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv4(x.float(), edge_index)
        x = gap(x, batch)

        # Fingerprints to tensor
        fp_list = []
        for smi in finger:
            mol = Chem.MolFromSmiles(smi)
            fp = []
            fp.extend(AllChem.GetMACCSKeysFingerprint(mol))
            fp.extend(AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1))
            fp.extend(GetPubChemFPs(mol))
            fp_list.append(fp)

        fp_array = np.array(fp_list, dtype=np.float32)
        fp_array = (fp_array - fp_array.min()) / (fp_array.max() - fp_array.min() + 1e-8)
        fp_tensor = torch.tensor(fp_array).to(x.device)

        # Encode fingerprints
        z, mu, logvar, recon_x = self.vae(fp_tensor)

        xc = torch.cat((x, z), dim=1)  

        # Clasificación
        y_binary = self.fc_binary(xc)
        y_binary = self.sigmoid(y_binary)

        return y_binary, recon_x, fp_tensor, mu, logvar


def train_fn(model, optimizer, train_loader, val_loader, test_loader, epochs,fold, beta=0.0001):  ## add validation argument

    model = model.to(device)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-3)
    my_loss_bin = BinaryClassLoss()
    earlystop = EarlyStopping(patience=patience, delta=delta)
    running_losses = []
    y_true_bin = np.array([])
    y_score_bin = np.zeros((0, label_num_binary), dtype=float)
    y_pred_bin = np.array([])

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for i, batch_data in enumerate(train_loader):
            batch_data = batch_data.to(device)
            optimizer.zero_grad()

            y_binary, recon_x, fp_tensor, mu, logvar = model(batch_data)
            label_bin_train = batch_data.y_bin.view(-1, 1).float()

            loss_class = my_loss_bin(y_binary, label_bin_train)
            loss_vae, loss_recon, loss_kl = vae_loss_function(recon_x, fp_tensor, mu, logvar)
            loss = loss_class + beta * loss_vae

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # 🔽 Guarda las métricas de validación
        with open("vae_metrics_train.csv", "a") as f:
            f.write(f"{epoch},train,{loss_recon:.4f},{loss_kl:.4f},{loss_vae.item():.4f},{fold}\n")
        model.eval()
        testing_loss = 0.0
        with torch.no_grad():
            for i, batch_data in enumerate(val_loader):
                batch_data = batch_data.to(device)
                y_binary, recon_x, fp_tensor, mu, logvar = model(batch_data)
                label_bin_test = batch_data.y_bin.view(-1, 1).float()
                loss_class = my_loss_bin(y_binary, label_bin_test)
                loss_vae, loss_recon, loss_kl = vae_loss_function(recon_x, fp_tensor, mu, logvar)

                loss = loss_class + beta * loss_vae
                testing_loss += loss.item()
                scheduler.step(loss)
            # 🔽 Guarda las métricas de validación
            with open("vae_metrics.csv", "a") as f:
                f.write(f"{epoch},val,{loss_recon:.4f},{loss_kl:.4f},{loss_vae.item():.4f}{fold}\n")
        print("epoch [%d] loss: %.6f testing_loss: %.6f " % (epoch + 1, running_loss / len(train_loader.dataset), testing_loss / len(val_loader.dataset)))
        running_losses.append(running_loss)
        train_epochs_loss.append(running_loss / len(train_loader.dataset))
        valid_epochs_loss.append(testing_loss / len(val_loader.dataset))
        earlystop(valid_epochs_loss[-1], model, file_path)
        if earlystop.early_stop:
            print("Early stopping\n")
            break

    ## Final evaluation: with test set
    pre_score_binary = np.zeros((0, label_num_binary), dtype=float)

    model.eval()
    with torch.no_grad():
        for i, batch_data in enumerate(test_loader):
            batch_data = batch_data.to(device)
            y_binary, _, _, _, _ = model(batch_data)
            label_bin_test = batch_data.y_bin
            pre_score_binary = np.vstack((pre_score_binary, y_binary.detach().cpu().numpy()))
            y_true_bin = np.hstack((y_true_bin, label_bin_test.cpu().numpy()))

        # 二分类
        pred_type_binary = (pre_score_binary > 0.6).astype(int)
        pred_type_binary = pred_type_binary.reshape(-1)
        accuracy = accuracy_score(y_true_bin, pred_type_binary)
        f1_micro = f1_score(y_true_bin, pred_type_binary, average="micro")
        f1_macro = f1_score(y_true_bin, pred_type_binary, average="macro")
        precision_micro = precision_score(y_true_bin, pred_type_binary, average="micro")
        precision_macro = precision_score(y_true_bin, pred_type_binary, average="macro")
        recall_micro = recall_score(y_true_bin, pred_type_binary, average="micro")
        recall_macro = recall_score(y_true_bin, pred_type_binary, average="macro")
        aupr = average_precision_score(y_true_bin, pre_score_binary)
        auroc = roc_auc_score(y_true_bin, pre_score_binary)
        print("acc", accuracy)
        print("f1_micro", f1_micro)
        print("f1_macro", f1_macro)
        print("precision_micro", precision_micro)
        print("precision_macro", precision_macro)
        print("recall_micro", recall_micro)
        print("recall_macro", recall_macro)
        print("AUPR:", aupr)
        print("AUROC:", auroc)

        return {
            "accuracy": accuracy,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "precision_micro": precision_micro,
            "precision_macro": precision_macro,
            "recall_micro": recall_micro,
            "recall_macro": recall_macro,
            "aupr": aupr,
            "auroc": auroc
        }

# ## Evaluation fn

# In[ ]:

# ## start training
def roc_aupr_score(y_true, y_score, average="macro"):  # y_true  形状为(n_samples,n_classes)二维数组 样本数 类别书 二进制编码表示  y_score形状为(n_samples,n_classes)二维数组 表示预测的概率分数
    def _binary_roc_aupr_score(y_true, y_score):  # macro 每个类别的分数平均值
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)

    def _average_binary_score(
            binary_metric, y_true, y_score, average
    ):  # y_true= y_one_hot
        if average == "binary":  # 仅对于二元分类问题
            return binary_metric(y_true, y_score)
        if average == "micro":  # 所有样本的总分数
            y_true = y_true.ravel()
            y_score = y_score.ravel()
        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_true.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()
            score[c] = binary_metric(y_true_c, y_score_c)
        return np.average(score)

    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)
# In[ ]:


if __name__ == '__main__':
    num_folds = 5
    with Pool(processes=num_folds) as pool:
        results = pool.map(cross_validate_kfolds, list(range(num_folds)))

    # guardamos CSV con fold y métricas
    df = pd.DataFrame(results)
    df.to_csv("kfold_results_vaeggt.csv", index=False)
    print("\nResultados guardados en kfold_results_vaeggt.csv")
