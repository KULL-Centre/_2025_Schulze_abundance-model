import glob
import os
import pathlib
import random
import subprocess
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from Bio.PDB.Polypeptide import index_to_one
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler

sys.path.append(f"{path_to_scripts}/scripts")

from helpers import (
    ds_train_val,
    ds_std_fixed_train_val,
    get_vamp_dataloader,
    train_val_split_ds
)

# add parser and parse input
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int)
parser.add_argument("--learning_rate", type=float)
parser.add_argument("--hidden_layer_1", type=int)
parser.add_argument("--hidden_layer_2", type=int)
parser.add_argument("--n_ensemble", type=int)
args = parser.parse_args()

# Set fixed seed
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

# Set constants
BATCH_SIZE_DS = args.batch_size
LEARNING_RATE_DS = args.learning_rate
NUM_ENSEMBLE = args.n_ensemble
PATIENCE = 3
EPOCHS_DS = 50 # upper limit to number of epochs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
hidden_size_1 = args.hidden_layer_1
hidden_size_2 = args.hidden_layer_2

# Set Downstream Network input features 

# define all possible features 
feature_list_all = [
     "wt_nlf","mt_nlf","wt_degron_score","mut_degron_score","WCN","rASA",
     "depth","PLDDT","Hbond_sum","dG_ESM_domain","degron_avg_domain"
]                 

# define features to use in this run
feature_list =  [
     "wt_degron_score","mut_degron_score","WCN","rASA",
     "depth","PLDDT","Hbond_sum","dG_ESM_domain","degron_avg_domain"
]

# set input layer size
ds_model_input_size = 60 + len(feature_list)

# define features to be normalised
# features not defined here will be used without any scaling or normalisation 
feature_norm_individual_list = [] 
feature_norm_global_list = ["masked-marginals","WCN","depth","Hbond_sum","dG_ESM_domain"]

def main():

    # Give path to data folder
    data_path = f"{path_to_data}/data"
    
    # Read DataFrame that includes all possible features
    df_vamp = pd.read_csv(f"{data_path}/train/downstream/df_vamp.csv")
    
    # Drop columns corresponding to features not present in feature_list 
    df_vamp = df_vamp.drop(columns=np.setdiff1d(feature_list_all,feature_list)) 
    
    # Checkpoint - save
    df_vamp.to_pickle(f"../output/vamp_train_val.pkl")

    # Checkpoint - load
    df_vamp = pd.read_pickle(f"../output/vamp_train_val.pkl")

    # Define loss function
    loss_ds = torch.nn.L1Loss()
    
    # Train downstream models
    print("Starting downstream model ensemble training")
    
    # Structure files to use in cross-validation
    pdbid_list = ['P60484','P51580','P11712','Q9NV35','P45381','O60260']

    # Do leave-one-out cross-validation
    for i in range(len(pdbid_list)):

        # Split into train and val
        pdbid_val = pdbid_list[i]

        # Do ensembling
        for j in range(NUM_ENSEMBLE):

            # Initialize model with fixed seed
            model_idx = j
            
            # Split into train and val data
            dataloader_train_ds, dataloader_val_ds = train_val_split_ds(
                df_vamp, [pdbid_val],  BATCH_SIZE_DS, DEVICE, MinMaxScaler(),
                feature_norm_individual_list, feature_norm_global_list, filter_val_data = True
            )
            
            # Train model
            print(f"Training leave-out-{pdbid_val} model: {j+1}/{NUM_ENSEMBLE}")
            print("Training downstream network only")
            
            ds_std_fixed_train_val(
                dataloader_train_ds,
                dataloader_val_ds,
                ds_model_input_size,
                loss_ds,
                LEARNING_RATE_DS,
                pdbid_list,
                pdbid_val,
                model_idx,
                EPOCHS_DS,
                PATIENCE,
                DEVICE,
                hidden_size_1,
                hidden_size_2
            )

        print(f"Finished training leave-out-{pdbid_val} model")
    
    print("Finished downstream model training")
            
if __name__ == "__main__":
    main()
