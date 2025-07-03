#!/usr/bin/env python
# # Import
from chemprop.features import morgan_binary_features_generator
# In[7]:


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
from torch_geometric.nn import global_mean_pool as gap

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
patience = 7
delta = 0
label_num_multi = 5
label_num_binary = 1
BATCH_SIZE=128

import multiprocessing as mp

if __name__ == '__main__':
    mp.set_start_method('spawn')

# ### Early stopping

# In[9]:


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
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
        self.len = len(x)
        self.x_data=x
        self.y_multi_data=y_multi
        self.y_binary_data=y_binary
    def __getitem__(self, index):
        return self.x_data[index], self.y_multi_data[index], self.y_binary_data[index]

    def __len__(self):
        return self.len


import torch.nn as nn


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
        self.fp_2_dim = 512
        self.fp_3_dim = 256
        # self.dropout_fpn = 0.2
        # self.hidden_dim = 512
        self.fp_type='mixed'

        if self.fp_type == 'mixed':
            self.fp_dim = 1489
        else:
            self.fp_dim = 1024

        self.fc1 = nn.Linear(self.fp_dim, self.fp_2_dim)
        self.fc2 = nn.Linear(self.fp_2_dim, self.fp_3_dim)
        self.act_func = nn.ReLU()
        # 二分类任务的输出层
        self.fc_binary = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes_binary)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x, finger,edge_index, batch = data.x, data.finger,data.edge_index, data.batch
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
        fp_array = np.array(fp_list, dtype=np.float32)
        fp_list = torch.tensor(fp_array).to(device)
        fpn_out = self.fc1(fp_list)
        fpn_out = self.act_func(fpn_out)
        fpn_out = self.fc2(fpn_out)
        fpn_out = self.act_func(fpn_out)
	# Aplica TransformerConv directamente a features originales
        x_trans = self.trans1(data.x.float(), data.edge_index)
        x_trans = F.relu(x_trans)
        x_trans = gap(x_trans, batch)
        # Fusiona ambas representaciones
        x = torch.cat([x_gat, x_trans], dim=1)  
        xc = torch.cat((x, fpn_out), dim=1)  

        # 二分类任务的预测
        y_binary = self.fc_binary(xc)
        y_binary = self.sigmoid(y_binary)

        return  y_binary


class MultiClassLoss(nn.Module):
    def __init__(self, gamma=2):
        super(MultiClassLoss, self).__init__()
        self.gamma = gamma

    def forward(self, preds, labels):
        labels = labels.view(-1, 1).type(torch.int64)  # [B * S, 1]
        preds = preds.view(-1, preds.size(-1))  # [B * S, C]

        preds_logsoft = F.log_softmax(preds, dim=1)  # 先softmax, 然后取log
        preds_softmax = torch.exp(preds_logsoft)  # softmax
        preds_softmax = preds_softmax.gather(1, labels)  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels)

        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = loss.mean()

        return loss

class BinaryClassLoss(nn.Module):
    def __init__(self):
        super(BinaryClassLoss, self).__init__()

    def forward(self, pred, target):

        target = target.view(-1, 1)
        criterion = nn.BCELoss()
        loss = criterion(pred, target.float())  # 将target转换为浮点数
        return loss


from torch.optim.lr_scheduler import CosineAnnealingLR
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
    model = MultiTaskModel(label_num_multi, label_num_binary)
    optimizer = Ranger(
        model.parameters(),
        lr=learn_rating,
        weight_decay=weight_decay_rate,
        betas=(0.95, 0.999),
        eps=1e-6
    )

    # entrenamiento + evaluación
    metrics = train_fn(model, optimizer, train_loader, val_loader, test_loader,300)
    metrics["fold"] = fold
    return metrics


def val():
    TRAIN_BATCH_SIZE = 128
    TEST_BATCH_SIZE = 128
    epochs=300
    model=MultiTaskModel(label_num_multi,label_num_binary)
    optimizer = Ranger(model.parameters(), lr=learn_rating, weight_decay=weight_decay_rate, betas=(0.95, 0.999),eps=1e-6)
    train_data = TestbedDataset(root='./feng/train/', path='train_graph_dataset.csv')
    test_data = TestbedDataset(root='./feng/test/', path='test_graph_dataset.csv')
    train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
    result_all,result_eve=train_fn(model,optimizer,train_loader,test_loader,epochs)
    return result_all,result_eve


def train_fn(model, optimizer, train_loader, val_loader, test_loader, epochs):
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
        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            outputs_bin_train = model(batch_data)
            label_bin_train = batch_data.y_bin
            loss_bin = my_loss_bin(outputs_bin_train, label_bin_train)
            optimizer.zero_grad()
            loss = loss_bin
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        testing_loss = 0.0
        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = batch_data.to(device)
                outputs_bin_test = model(batch_data)
                label_bin_test = batch_data.y_bin
                loss_bin = my_loss_bin(outputs_bin_test, label_bin_test)
                loss = loss_bin
                testing_loss += loss.item()

        print("epoch [%d] loss: %.6f testing_loss: %.6f " % (epoch + 1, running_loss / len(train_loader.dataset), testing_loss / len(val_loader.dataset)))
        running_losses.append(running_loss)
        train_epochs_loss.append(running_loss / len(train_loader.dataset))
        valid_epochs_loss.append(testing_loss / len(val_loader.dataset))
        earlystop(valid_epochs_loss[-1], model, file_path)
        if earlystop.early_stop:
            print("Early stopping\n")
            break

    pre_score_binary = np.zeros((0, label_num_binary), dtype=float)

    model.eval()
    with torch.no_grad():
        for batch_data in test_loader:
            batch_data = batch_data.to(device)
            outputs_bin = model(batch_data)
            label_bin_test = batch_data.y_bin
            pre_score_binary = np.vstack((pre_score_binary, torch.sigmoid(outputs_bin).cpu().numpy()))
            y_true_bin = np.hstack((y_true_bin, label_bin_test.cpu().numpy()))

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


def roc_aupr_score(y_true, y_score, average="macro"):
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)

    def _average_binary_score(
            binary_metric, y_true, y_score, average
    ):  # y_true= y_one_hot
        if average == "binary":
            return binary_metric(y_true, y_score)
        if average == "micro":
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
def evaluate(y_pred, y_score, y_true, label_num):
    all_eval_type = 11
    result_all = np.zeros((all_eval_type, 1), dtype=float)
    each_eval_type = 6
    result_eve = np.zeros((label_num, each_eval_type), dtype=float)
    y_one_hot = label_binarize(y_true, classes=range(label_num))
    pred_one_hot = label_binarize(y_pred, classes=range(label_num))
    result_all[0] = accuracy_score(y_true, y_pred)
    print('acc',result_all[0])
    result_all[1] = roc_aupr_score(y_one_hot, y_score, average="micro")
    print('aupr_micro', result_all[1])
    result_all[2] = roc_aupr_score(y_one_hot, y_score, average="macro")
    print('aupr_macro', result_all[2])
    result_all[3] = roc_auc_score(y_one_hot, y_score, average="micro")
    print('auc_micro', result_all[3])
    result_all[4] = roc_auc_score(y_one_hot, y_score, average="macro")
    print('auc_macro', result_all[4])
    result_all[5] = f1_score(y_true, y_pred, average="micro")
    print('f1_micro', result_all[5])
    result_all[6] = f1_score(y_true, y_pred, average="macro")
    print('f1_macro', result_all[6])
    result_all[7] = precision_score(y_true, y_pred, average="micro")
    print('precision_micro', result_all[7])
    result_all[8] = precision_score(y_true, y_pred, average="macro")
    print('precision_macro', result_all[8])
    result_all[9] = recall_score(y_true, y_pred, average="micro")
    print('recall_micro', result_all[9])
    result_all[10] = recall_score(y_true, y_pred, average="macro")
    print('recall_macro', result_all[10])
    for i in range(label_num_multi):
        result_eve[i, 0] = accuracy_score(
            y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel()
        )
        result_eve[i, 1] = roc_aupr_score(
            y_one_hot.take([i], axis=1).ravel(),
            pred_one_hot.take([i], axis=1).ravel(),
            average=None,
        )
        result_eve[i, 2] = roc_auc_score(
            y_one_hot.take([i], axis=1).ravel(),
            pred_one_hot.take([i], axis=1).ravel(),
            average=None,
        )
        result_eve[i, 3] = f1_score(
            y_one_hot.take([i], axis=1).ravel(),
            pred_one_hot.take([i], axis=1).ravel(),
            average="binary",
        )
        result_eve[i, 4] = precision_score(
            y_one_hot.take([i], axis=1).ravel(),
            pred_one_hot.take([i], axis=1).ravel(),
            average="binary",
        )
        result_eve[i, 5] = recall_score(
            y_one_hot.take([i], axis=1).ravel(),
            pred_one_hot.take([i], axis=1).ravel(),
            average="binary",
        )
    return [result_all, result_eve]

def save_result(result_type, result):
    index = ['accuracy', 'aupr_micro', 'aupr_macro', 'auc_micro', 'auc_macro', 'f1_micro', 'f1_macro',
             'precision_micro', 'precision_macro', 'recall_micro', 'recall_macro']

    if result_type == 'all':
        all_ = pd.DataFrame(result, index=index)
        all_.to_csv('./results_all_multi_gatgcn.csv')
    else:
        each = pd.DataFrame(result)
        each.to_csv('./results_each_multi_gatgcn.csv', index=False)



if __name__ == '__main__':

    num_folds = 5
    with Pool(processes=num_folds) as pool:
        results = pool.map(cross_validate_kfolds, list(range(num_folds)))

    # guardamos CSV con fold y métricas
    df = pd.DataFrame(results)
    df.to_csv("kfold_results_gatgcnTrans.csv", index=False)
    print("\nResultados guardados en kfold_results_gatgcnTrans.csv")

