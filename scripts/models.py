import glob
import os
import random
from typing import Callable, List, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class DownstreamModel1(torch.nn.Module):
    """
    Downstream dense (abundance) neural network 
    with single hidden layer with number of nodes
    given by hidden_size_1.
    """
    def __init__(self, input_size, hidden_size_1):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1

        # Model
        self.lin1 = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, hidden_size_1), 
            torch.nn.BatchNorm1d(hidden_size_1),
            torch.nn.LeakyReLU(),
        )
        self.lin2 = torch.nn.Linear(in_features=hidden_size_1, out_features=1)

    # Forward
    def forward(self, x):
        x = self.lin1(x)
        x = self.lin2(x)
        return x
    
    
class DownstreamModel2(torch.nn.Module):
    """
    Downstream dense (abundance) neural network
    with two hidden layers with numbers of nodes
    given by hidden_size_1 and hidden_size_2.
    """
    def __init__(self, input_size, hidden_size_1, hidden_size_2):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2

        # Model
        self.lin1 = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, hidden_size_1), 
            torch.nn.BatchNorm1d(hidden_size_1),
            torch.nn.LeakyReLU(),
        )
        self.lin2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_size_1, hidden_size_2),
            torch.nn.BatchNorm1d(hidden_size_2),
            torch.nn.LeakyReLU(),
        )
        self.lin3 = torch.nn.Linear(in_features=hidden_size_2, out_features=1)

    # Forward
    def forward(self, x):
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)
        return x


class StandardCurveModel(torch.nn.Module): 
    """
    Standard curve model. 
    """
    def __init__(self):
        super().__init__()
        
        # Model
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(1,2),
            torch.nn.Tanh(),           
            torch.nn.Linear(2,1),      
        )
        
    # Forward   
    def forward(self,x):
        x = self.seq(x)
        return x 
    
    
class VAMPDataset(Dataset):   
    """
    Custom Dataset class for input data. 
    """
    def __init__(self,df,transformer=None):
        self.df = df
        self.transformer = transformer

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        if self.transformer:
            sample = self.transformer(sample)
        return sample


class VAMPToTensor:           
    """
    To-tensor transformer for input dataframe.

    Parameters
    ----------
    DEVICE: str
        Either "cuda" (gpu) or "cpu". Is set-able.
    """

    def __init__(self, flag, DEVICE):   
        self.device = DEVICE          
        self.flag = flag              
        
    def __call__(self, sample):
        
        # get esm score column names
        if "masked_marginal_idx_0" in sample.index:
            esm_scores_column_names = [f"masked_marginal_idx_{i}" for i in range(20)]

        # list esm scores
        esm_scores = list(sample[esm_scores_column_names].values)
        
        # create one-hot enconding of WT and variant
        wt_onehot = np.zeros(20)
        wt_onehot[sample["wt_idx"]] = 1.0
        mt_onehot = np.zeros(20)
        mt_onehot[sample["mt_idx"]] = 1.0
        
        # list downstream network (abundance network) input features 
        x_ds_list = [esm_scores, wt_onehot, mt_onehot]
        
        # list all possible features 
        feature_list_all = np.array(
            ["wt_nlf","mt_nlf","wt_degron_score","mut_degron_score","WCN","rASA",
             "depth","PLDDT","Hbond_sum","dG_ESM_domain","degron_avg_domain"]
        )
        
        # check which features are present in the input dataset
        feature_list = feature_list_all[np.isin(feature_list_all, sample.index)]
        
        # add features to downstream network input
        for feature in feature_list:
            x = [sample[feature]]
            x_ds_list.append(x)
        x_ds = np.hstack(x_ds_list)    
        
        # get protein and variant data 
        pdbid = [sample["pdbid"]]
        chainid = [sample["chainid"]]
        variant = [sample["variant"]]
        
        if self.flag != "pred":
            vamp_score = [sample["score"]]
            return {
                "pdbid": pdbid,
                "chainid": chainid,
                "variant": variant,
                "x_ds": torch.tensor(x_ds, dtype=torch.float32).to(self.device),
                "score": torch.tensor(vamp_score, dtype=torch.float32).to(self.device),
            }
        else:
            return {
                "pdbid": pdbid,
                "chainid": chainid,
                "variant": variant,
                "x_ds": torch.tensor(x_ds, dtype=torch.float32).to(self.device),
            }

    def collate_multi(self, batch):
        """
        Collate method used by the dataloader to collate a
        batch of multi data.
        """
        pdbid_batch = [b["pdbid"] for b in batch]
        chainid_batch = [b["chainid"] for b in batch]
        variant_batch = [b["variant"] for b in batch]
        x_ds_batch = torch.cat([torch.unsqueeze(b["x_ds"], 0) for b in batch], dim=0)

        if self.flag != "pred":
            vamp_score_batch = torch.cat(
                [torch.unsqueeze(b["score"], 0) for b in batch], dim=0
            )
            return pdbid_batch, chainid_batch, variant_batch, x_ds_batch, vamp_score_batch

        else:
            return pdbid_batch, chainid_batch, variant_batch, x_ds_batch
