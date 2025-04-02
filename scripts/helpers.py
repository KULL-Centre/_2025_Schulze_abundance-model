import datetime
import glob
import itertools
import os
import pickle
import random
import glob
import sys
from typing import Dict, List, Union
import shutil

import numpy as np
import pandas as pd
import pytz
import torch
from Bio.PDB.Polypeptide import index_to_one, one_to_index
from scipy.stats import pearsonr, spearmanr
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, Dataset

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.neural_network import MLPRegressor

from models import (
    VAMPDataset,
    VAMPToTensor,
    DownstreamModel1,
    DownstreamModel2,
    StandardCurveModel,
)

def get_vamp_dataloader(vamp_data, data_type, BATCH_SIZE, DEVICE): 
    """
    Returns a dataloader based on input data and data type.
    """
    # Define ddG data set
    vamp_dataset = VAMPDataset(vamp_data, transformer=VAMPToTensor(data_type, DEVICE))       
                                                                                             
    # Define case-dependent data loader parameters
    shuffle = False
    if data_type == "train":
        BATCH_SIZE = BATCH_SIZE
        shuffle = True
        drop_last = True
    elif data_type == "train_val_std":
        BATCH_SIZE = BATCH_SIZE
        shuffle = True
        drop_last = True
    elif data_type == "val":
        BATCH_SIZE = BATCH_SIZE
        shuffle = False
        drop_last = True    
    elif data_type == "test" or data_type == "pred":
        BATCH_SIZE = 100  # Worst-case max gpu memory on small machine
        shuffle = False
        drop_last = False

    # Define data loader
    vamp_dataloader = DataLoader(
        vamp_dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=VAMPToTensor(data_type, DEVICE).collate_multi,
    )

    return vamp_dataloader


def filter_shuffle_val_ds(vamp_data_val, pdbid_val_list, random_state):
    """
    Removes variant rows from input dataframe that were already used for 
    the initial fitting of the standard curve network, and shuffles rows. 
    """
    # works only for single val protein
    assert len(pdbid_val_list) == 1
    
    # define dict
    pdbid_protname_dict = {
    'P60484':'PTEN', 'P51580':'TPMT', 'P11712':'CYP2C9',
    'Q9NV35':'NUDT15', 'P45381':'ASPA', 'O60260':'PRKN'
    }
    
    # get dict with names of variants used for initial standard curve network fitting
    init_fit_var_names_dict = pd.read_pickle(f"{path_to_data}/data/low_TP_variant_names.pkl')
    
    # get arr with variant names for validation protein
    init_fit_var_names_arr = init_fit_var_names_dict[pdbid_protname_dict[pdbid_val_list[0]]]
    
    # remove variants used for initial standard curve network fitting
    vamp_data_val = vamp_data_val.iloc[~np.isin(vamp_data_val.variant.values, init_fit_var_names_arr)]
    
    # shuffle data points randomly along index 
    vamp_data_val_shuffled = vamp_data_val.sample(frac=1, replace=False, random_state=random_state, axis=0)
    
    return vamp_data_val_shuffled


def filter_val_ds(vamp_data_val, pdbid_val_list):
    """
    Removes variant rows from input dataframe that were already used for 
    the initial fitting of the standard curve network.
    """
    # works only for single val protein
    assert len(pdbid_val_list) == 1
    
    # define dict
    pdbid_protname_dict = {
    'P60484':'PTEN', 'P51580':'TPMT', 'P11712':'CYP2C9',
    'Q9NV35':'NUDT15', 'P45381':'ASPA', 'O60260':'PRKN'
    }
    
    # get dict with names of variants used for initial standard curve network fitting
    init_fit_var_names_dict = pd.read_pickle(f"{path_to_data}/data/low_TP_variant_names.pkl')    
    
    # get arr with variant names for validation protein
    init_fit_var_names_arr = init_fit_var_names_dict[pdbid_protname_dict[pdbid_val_list[0]]]
    
    # remove variants used for initial standard curve network fitting
    vamp_data_val = vamp_data_val.iloc[~np.isin(vamp_data_val.variant.values, init_fit_var_names_arr)]
    
    return vamp_data_val


def feature_scale_transform(df, feature, feature_transformer):
    """
    Transforms input feature found in df with feature_transformer
    and returns df with updated feature values. 
    """
    feature_arr = feature_transformer.transform(df[feature].values.reshape(-1,1)).flatten()
    df = df.drop(columns=[feature])
    df[feature] = feature_arr
    return df


def feature_norm_individual(vamp_data, feature_scaler, feature_list):
    """
    Normalise ESM-IF scores and features defined in feature_list using feature_scaler. 
    Normalisation is done per protein defined in pdbid_list. 
    """
    # initialise
    df_list = []

    # loop over proteins
    pdbid_list = ['P60484','P51580','P11712','Q9NV35','P45381','O60260']

    for pdbid in pdbid_list:

        # slice dataframe to contain data for only one protein
        df = vamp_data[vamp_data["pdbid"].isin([pdbid])]
        
        if "masked-marginals" in feature_list:
        
            # fit scaler for ESM-IF data and transform
            marginals_transformer = feature_scaler.fit(df["masked-marginals"].values.reshape(-1,1))
            for i in range(20):
                df = feature_scale_transform(df, f"masked_marginal_idx_{i}", marginals_transformer)

        # transform all other features
        for feature in np.setdiff1d(feature_list,["masked-marginals"]):
            
            # fit feature_scaler for specific feature
            feature_transformer = feature_scaler.fit(df[feature].values.reshape(-1,1))
            
            # transform this feature
            df = feature_scale_transform(df, feature, feature_transformer)

        df_list.append(df)
        
    # concatenate individual dataframes    
    df = pd.concat(df_list)
    
    return df


def features_norm_global(vamp_data_train, vamp_data_val, feature_scaler, feature_list):
    """
    Normalise ESM-IF scores and features defined in feature_list using feature_scaler. 
    Normalisation is done globally, i.e. by fitting to the training data and applying the 
    same normalisation or scaling to both the training and the validation data. 
    """
    # transform features in feature_list
    for feature in np.setdiff1d(feature_list,["masked-marginals"]):

        # fit on training data
        feature_transformer = feature_scaler.fit(vamp_data_train[feature].values.reshape(-1,1))

        # transform training and validation data
        vamp_data_train = feature_scale_transform(vamp_data_train, feature, feature_transformer)
        vamp_data_val = feature_scale_transform(vamp_data_val, feature, feature_transformer)

    if "masked-marginals" in feature_list:    
        
        # transform remaining features, i.e. ESM-IF masked marginals
        marginals_transformer = feature_scaler.fit(vamp_data_train["masked-marginals"].values.reshape(-1,1))
        for i in range(20):

            vamp_data_train = feature_scale_transform(vamp_data_train, f"masked_marginal_idx_{i}", marginals_transformer)
            vamp_data_val = feature_scale_transform(vamp_data_val, f"masked_marginal_idx_{i}", marginals_transformer)

        # print attributes of feature_scaler 
        #print("marginals_transformer attributes:")
        #print(vars(marginals_transformer))
    
    return vamp_data_train, vamp_data_val


def train_val_split_ds(vamp_data, pdbid_val_list, BATCH_SIZE, DEVICE, feature_scaler, feature_norm_individual_list, feature_norm_global_list, filter_val_data):  
    """
    Function to perform training and validation split of vamp_data dataframe.
    Use function when std curves are not included or included but not trained. 
    """
    # norm features for individual proteins
    if len(feature_norm_individual_list) > 0:
        vamp_data = feature_norm_individual(
            vamp_data, feature_scaler, feature_norm_individual_list
        )  
    
    parsed_pdb_filenames = vamp_data["pdbid"].unique()
    filenames_val = np.array(pdbid_val_list)
    filenames_train = np.setdiff1d(parsed_pdb_filenames, filenames_val)

    vamp_data_train = vamp_data[vamp_data["pdbid"].isin(filenames_train)]
    vamp_data_val = vamp_data[vamp_data["pdbid"].isin(filenames_val)]
    
    # norm features across proteins
    if len(feature_norm_global_list) > 0:
        vamp_data_train, vamp_data_val = features_norm_global(
            vamp_data_train, vamp_data_val, feature_scaler, feature_norm_global_list
        )
    
    # remove validation data points that were used for standard curve fitting
    if filter_val_data == True:
        vamp_data_val = filter_val_ds(vamp_data_val, pdbid_val_list)
    
    # save scaled/normalised data
    vamp_data_train.to_csv(f"../output/vamp_data_train_{pdbid_val_list[0]}.csv")
    vamp_data_val.to_csv(f"../output/vamp_data_val_{pdbid_val_list[0]}.csv")    
    
    dataloader_train = get_vamp_dataloader(vamp_data_train, "train", BATCH_SIZE, DEVICE)  
    dataloader_val = get_vamp_dataloader(vamp_data_val, "val", BATCH_SIZE, DEVICE) 

    print(f"Training data set includes {len(filenames_train)} pdbs with "
          f"{len(vamp_data_train)} mutations."
         )
    print(f"Training PDBs are: {filenames_train}.")
    print(f"Validation data set includes {len(filenames_val)} pdbs with "
          f"{len(vamp_data_val)} mutations."
         )
    print(f"Validation PDBs are: {filenames_val}.")
    
    return dataloader_train, dataloader_val


def val_split_all_ds(vamp_data, pdbid_val_list, BATCH_SIZE, DEVICE, feature_scaler, feature_norm_individual_list, feature_norm_global_list, filter_val_data):  
    """
    Function to perform training and validation split of vamp_data dataframe.
    Uses data_type=test to not drop samples in last batch (for testing performance
    on all variants in dataset).
    """
    # norm features for individual proteins
    if len(feature_norm_individual_list) > 0:
        vamp_data = feature_norm_individual(
            vamp_data, feature_scaler, feature_norm_individual_list
        )  
    
    parsed_pdb_filenames = vamp_data["pdbid"].unique()
    filenames_val = np.array(pdbid_val_list)
    filenames_train = np.setdiff1d(parsed_pdb_filenames, filenames_val)

    vamp_data_train = vamp_data[vamp_data["pdbid"].isin(filenames_train)]
    vamp_data_val = vamp_data[vamp_data["pdbid"].isin(filenames_val)]
    
    # norm features across proteins
    if len(feature_norm_global_list) > 0:
        vamp_data_train, vamp_data_val = features_norm_global(
            vamp_data_train, vamp_data_val, feature_scaler, feature_norm_global_list
        )
    
    # remove validation data points that were used for standard curve fitting
    if filter_val_data == True:
        vamp_data_val = filter_val_ds(vamp_data_val, pdbid_val_list)
    
    # save scaled/normalised data
    vamp_data_val.to_csv(f"../output/vamp_data_val_all_{pdbid_val_list[0]}.csv")    
    
    # make dataloader
    dataloader_val = get_vamp_dataloader(vamp_data_val, "test", BATCH_SIZE, DEVICE) 

    print(f"Validation data set includes {len(filenames_val)} pdbs with "
          f"{len(vamp_data_val)} mutations."
         )
    print(f"Validation PDBs are: {filenames_val}.")
    
    return dataloader_val


def train_val_split_ds_std(vamp_data, pdbid_val_list, val_split_seed, BATCH_SIZE, DEVICE, feature_scaler, feature_norm_individual_list, feature_norm_global_list):
    """
    Function to perform training and validation split of vamp_data dataframe.
    Use function when std curves are included and trained. 
    Puts aside 20% of validation protein data points to train validation protein
    standard curve while keeping 80% of validation protein datapoints for actual validation. 
    """
    # norm features for individual proteins
    if len(feature_norm_individual_list) > 0:
        vamp_data = feature_norm_individual(
            vamp_data, feature_scaler, feature_norm_individual_list
        ) 
    
    # define training and validation protein lists
    parsed_pdb_filenames = vamp_data["pdbid"].unique()
    filenames_val = np.array(pdbid_val_list)
    filenames_train = np.setdiff1d(parsed_pdb_filenames, filenames_val)

    # split data into train and val
    vamp_data_train = vamp_data[vamp_data["pdbid"].isin(filenames_train)]
    vamp_data_val = vamp_data[vamp_data["pdbid"].isin(filenames_val)]

    # norm features across proteins
    if len(feature_norm_global_list) > 0:
        vamp_data_train, vamp_data_val = features_norm_global(
            vamp_data_train, vamp_data_val, feature_scaler, feature_norm_global_list
        )
    
    # save scaled/normalised data
    vamp_data_train.to_csv("../output/vamp_data_train.csv")
    vamp_data_val.to_csv("../output/vamp_data_val.csv")
    
    # filter and shuffle validation examples
    vamp_data_val_shuffled = filter_shuffle_val_ds(vamp_data_val, pdbid_val_list, val_split_seed)
    
    # set number of training examples for training validation protein standard curve
    train_val_std_examples = int(np.around(len(vamp_data_val_shuffled) * 0.2))

    # split validation data into training and proper validation data
    vamp_data_train_val_std = vamp_data_val_shuffled.iloc[:train_val_std_examples]
    vamp_data_prop_val = vamp_data_val_shuffled.iloc[train_val_std_examples:]

    # repeat train_val data five times to see more examples in every epoch of the training
    vamp_data_train_val_std_repeat = pd.concat([vamp_data_train_val_std] * int(1 / 0.2))
    
    # check that there's no overlap between variants in validation sets
    assert len(np.intersect1d(vamp_data_train_val_std_repeat.variant.values, vamp_data_prop_val.variant.values)) == 0

    # get DataLoader objects
    dataloader_train = get_vamp_dataloader(vamp_data_train, "train", BATCH_SIZE, DEVICE)
    dataloader_train_val_std = get_vamp_dataloader(vamp_data_train_val_std_repeat, "train_val_std", BATCH_SIZE, DEVICE)
    dataloader_val = get_vamp_dataloader(vamp_data_prop_val, "val", BATCH_SIZE, DEVICE)

    # print data set info
    print(
        f"Training data set includes {len(filenames_train)} pdbs with "
        f"{len(vamp_data_train)} mutations."
    )
    print(f"Training PDBs are: {filenames_train}.")
    print(
        f"Validation standard curve training data set includes {len(filenames_val)} pdbs with "
        f"{len(vamp_data_train_val_std_repeat)} mutations."
    )
    print(
        f"Validation data set includes {len(filenames_val)} pdbs with "
        f"{len(vamp_data_prop_val)} mutations."
    )
    print(f"Validation PDBs are: {filenames_val}.")

    return dataloader_train, dataloader_train_val_std, dataloader_val


def init_lin_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)

        
def init_standard_curve_network(pdbid, DEVICE):
    """
    Initialise standard curve network with weights and biases from 
    previous fitting with MLPRegressor class from sklearn. 
    """
    std_regr_dict = pd.read_pickle(f"{path_to_data}/data/MLPreg.pkl')               

    pdbid_protname_dict = {
    'P60484':'PTEN', 'P51580':'TPMT', 'P11712':'CYP2C9',
    'Q9NV35':'NUDT15', 'P45381':'ASPA', 'O60260':'PRKN'
    }
    
    prot = pdbid_protname_dict[pdbid]
    
    w_init = std_regr_dict[prot][0].coefs_ # list of model weights as output by the sklearn MLPRegressor class 
    b_init = std_regr_dict[prot][0].intercepts_ # list of model biases as output by the sklearn MLPRegressor class
    
    # initialise model
    standard_curve_net = StandardCurveModel().to(DEVICE)
    
    # set model params to params of previous fit
    standard_curve_net.state_dict()["seq.0.weight"][:] = torch.Tensor(w_init[0].T)
    standard_curve_net.state_dict()["seq.0.bias"][:] = torch.Tensor(b_init[0])
    standard_curve_net.state_dict()["seq.2.weight"][:] = torch.Tensor(w_init[1].T)
    standard_curve_net.state_dict()["seq.2.bias"][:] = torch.Tensor(b_init[1])
        
    return standard_curve_net    


def ds_train_val( 
    dataloader_train,
    dataloader_val,
    ds_model_input_size,
    loss_func,
    LEARNING_RATE_DS,
    pdbid_val,
    model_idx,
    res_split_idx, 
    EPOCHS,
    PATIENCE, 
    DEVICE,
    hidden_size_1,
    hidden_size_2
):
    """
    Function to train and validate downstream/abundance network. 
    Use when training with no standard curves. 
    """    
    # Set seed
    np.random.seed(model_idx)
    random.seed(model_idx)
    torch.manual_seed(model_idx)
    torch.cuda.manual_seed(model_idx)
    torch.cuda.manual_seed_all(model_idx)
    
    # Initialise downstream model
    if hidden_size_2 > 0:
        ds_model_net = DownstreamModel2(ds_model_input_size, hidden_size_1, hidden_size_2).to(DEVICE)
    else:
        ds_model_net = DownstreamModel1(ds_model_input_size, hidden_size_1).to(DEVICE)
    torch.manual_seed(seed=model_idx)
    torch.cuda.manual_seed(seed=model_idx)
    ds_model_net.apply(init_lin_weights)
    optimizer = torch.optim.Adam(ds_model_net.parameters(), lr=LEARNING_RATE_DS)
    
    # Define parameters for early stopping
    current_best_epoch_idx = -1
    current_best_loss_val = 1e4
    patience = 0
    
    # Define name of output directory
    if isinstance(res_split_idx, int):
        outpath = f"../output/ds_models/train_val_{pdbid_val}"  
        ds_model_idx = res_split_idx
    else:    
        outpath = f"../output/ds_models/leave_out_{pdbid_val}"       
        ds_model_idx = model_idx
        
    # Create output directory if it doesn't already exist
    if not os.path.isdir(f'{outpath}'):
            os.mkdir(f'{outpath}')
    if not os.path.isdir(f'{outpath}/ds_model_{ds_model_idx}'):
            os.mkdir(f'{outpath}/ds_model_{ds_model_idx}')
    if not os.path.isdir(f'{outpath}/ds_model_{ds_model_idx}/no_std'):
            os.mkdir(f'{outpath}/ds_model_{ds_model_idx}/no_std')
    
    # Initialize lists
    train_loss_list = []
    val_loss_list = []
    pearson_r_list = []
    val_vamp_list = []
    val_vamp_pred_list = []
    val_variant_list = []
    
    # Run training and validation over EPOCHS
    for epoch in range(EPOCHS):

        print(f"Epoch: {epoch+1}/{EPOCHS}")

        # Initialize
        train_loss_batch_list = torch.empty(0, dtype=torch.float32).to(DEVICE)
        val_loss_batch_list = torch.empty(0, dtype=torch.float32).to(DEVICE)
        val_vamp = torch.empty(0, dtype=torch.float32).to(DEVICE)
        val_vamp_pred = torch.empty(0, dtype=torch.float32).to(DEVICE)
        val_variant = []

        # Train loop
        ds_model_net.train()

        for _, _, variant_batch, x_ds_batch, vamp_batch in dataloader_train:
            
            # Initialize optimizer
            optimizer.zero_grad()

            # Compute predictions
            vamp_batch_pred = ds_model_net(x_ds_batch)
            
            loss_batch = loss_func(vamp_batch_pred, vamp_batch)
            loss_batch.backward()
            optimizer.step()

            # Append to epoch
            train_loss_batch_list = torch.cat(
                (train_loss_batch_list, loss_batch.detach().reshape(-1))
            )

        # Val loop
        ds_model_net.eval()
        with torch.no_grad():

            for _, _, val_variant_batch, val_x_ds_batch, val_vamp_batch in dataloader_val:

                # Compute predictions
                val_vamp_pred_batch = ds_model_net(val_x_ds_batch)
                
                val_loss_batch = loss_func(
                    val_vamp_pred_batch, val_vamp_batch
                )

                # Append to epoch
                val_loss_batch_list = torch.cat(
                    (val_loss_batch_list, val_loss_batch.reshape(-1))
                )
                val_vamp = torch.cat(
                    (val_vamp, val_vamp_batch.reshape(-1)), 0
                )
                val_vamp_pred = torch.cat(
                    (val_vamp_pred, val_vamp_pred_batch.reshape(-1)), 0
                )

                val_variant.append(np.concatenate(val_variant_batch))

        # Compute epoch metrics
        train_loss_list.append(train_loss_batch_list.mean().cpu().item())
        val_loss_list.append(val_loss_batch_list.mean().cpu().item())
        val_vamp = val_vamp.detach().cpu()
        val_vamp_pred = val_vamp_pred.detach().cpu()
        pearson_r_list.append(pearsonr(val_vamp_pred.numpy(), val_vamp.numpy())[0])

        val_vamp_list.append(val_vamp.numpy())
        val_vamp_pred_list.append(val_vamp_pred.numpy())
        val_variant_list.append(np.concatenate(val_variant))     
        
        # Save from model current epoch
        torch.save(ds_model_net.state_dict(),f"{outpath}/ds_model_{ds_model_idx}/no_std/model_{epoch}.pt")
        
        # Compare loss to current best loss and modify patience
        loss_val = val_loss_batch_list.mean().cpu().item()
        if loss_val < current_best_loss_val:
            current_best_loss_val = loss_val
            current_best_epoch_idx = epoch
            patience = 0
        else:
            patience += 1
            
        # Early stopping
        if patience == PATIENCE:
            break
      
    # Save results evaluated over all epochs
    lc_results = (pearson_r_list, val_loss_list, train_loss_list, val_vamp_list, 
                  val_vamp_pred_list, val_variant_list, pdbid_val, model_idx)
    with open(f"{outpath}/ds_model_{ds_model_idx}/no_std/lc_results.pkl","wb",) as f:
        pickle.dump(lc_results, f)
    
    # Copy best model
    shutil.copyfile(f"{outpath}/ds_model_{ds_model_idx}/no_std/model_{current_best_epoch_idx}.pt",
                    f"{outpath}/ds_model_{ds_model_idx}/no_std/model.pt")
    
    # Delete saved ds_model_net files that are not from best epoch
    ds_model_net_files = glob.glob(f'{outpath}/ds_model_{model_idx}/no_std/model_*')
    for f in ds_model_net_files:
        os.remove(f)
    
    # Print
    print(
        f"Best epoch: {current_best_epoch_idx + 1} with validation loss: "
        f"{current_best_loss_val:5.3f}"
    )
    

def ds_std_fixed_train_val(
    dataloader_train,
    dataloader_val,
    ds_model_input_size,
    loss_func,
    LEARNING_RATE_DS, 
    pdbid_list,
    pdbid_val,
    model_idx,
    EPOCHS,
    PATIENCE,
    DEVICE,
    hidden_size_1,
    hidden_size_2
):
    """
    Function to train and validate downstream/abundance network. 
    Use when training with fixed standard curves. 
    """
    # Set seed
    np.random.seed(model_idx)
    random.seed(model_idx)
    torch.manual_seed(model_idx)
    torch.cuda.manual_seed(model_idx)
    torch.cuda.manual_seed_all(model_idx) 
    
    # Initialise downstream network with either one or two hidden layers
    if hidden_size_2 > 0:
        ds_model_net = DownstreamModel2(ds_model_input_size, hidden_size_1, hidden_size_2).to(DEVICE)
    else:
        ds_model_net = DownstreamModel1(ds_model_input_size, hidden_size_1).to(DEVICE)

    # Intitialise model with random weights
    torch.manual_seed(seed=model_idx)
    torch.cuda.manual_seed(seed=model_idx)
    ds_model_net.apply(init_lin_weights)
    
    # Initialise standard curve networks
    standard_curve_network_dict = {}
    for pdbid in pdbid_list:
        standard_curve_network_dict[pdbid] = init_standard_curve_network(pdbid, DEVICE)
    
    # Define optimiser for ds_net training
    optimizer_train = torch.optim.Adam(ds_model_net.parameters(), lr=LEARNING_RATE_DS)
    
    # Define parameters for early stopping
    current_best_epoch_idx = -1
    current_best_loss_val = 1e4
    patience = 0
    
    # Define name of output directory    
    outdir = "std_fixed"
    outpath = f"../output/ds_models/leave_out_{pdbid_val}"
    
    # Create output directory
    if not os.path.isdir(outpath):
            os.mkdir(outpath)
    if not os.path.isdir(f'{outpath}/ds_model_{model_idx}'):
            os.mkdir(f'{outpath}/ds_model_{model_idx}')
    if not os.path.isdir(f'{outpath}/ds_model_{model_idx}/{outdir}'):
            os.mkdir(f'{outpath}/ds_model_{model_idx}/{outdir}')
    
    # Initialize lists
    train_loss_list = []
    val_loss_list = []
    pearson_r_list = []

    val_vamp_list = []
    val_vamp_pred_list = []
    val_vamp_std_pred_list = []
    val_variant_list = []
    
    # Run training and validation over EPOCHS
    for epoch in range(EPOCHS):

        print(f"Epoch: {epoch+1}/{EPOCHS}")

        # Initialize
        train_loss_batch_list = torch.empty(0, dtype=torch.float32).to(DEVICE)
        val_loss_batch_list = torch.empty(0, dtype=torch.float32).to(DEVICE)
        val_vamp = torch.empty(0, dtype=torch.float32).to(DEVICE)
        val_vamp_pred = torch.empty(0, dtype=torch.float32).to(DEVICE)
        val_vamp_std_pred = torch.empty(0, dtype=torch.float32).to(DEVICE)
        val_vamp = torch.empty(0, dtype=torch.float32).to(DEVICE)
        val_variant = []
        
        # Train loop
        ds_model_net.train()  

        for pdbid_batch, _, _, x_ds_batch, vamp_batch in dataloader_train:
            
            # Initialize optimizer
            optimizer_train.zero_grad()

            # Compute predictions 
            vamp_batch_pred = ds_model_net(x_ds_batch)
            
            # Transform data with standard curve specific to each protein
            vamp_std_batch_pred = torch.empty(len(pdbid_batch), dtype=torch.float32, requires_grad=True).to(DEVICE)
            for b in range(len(pdbid_batch)): 
                vamp_std_batch_pred[b] = standard_curve_network_dict[pdbid_batch[b][0]](vamp_batch_pred[b])            
            
            # Calculate loss
            loss_batch = loss_func(vamp_std_batch_pred.reshape(-1,1), vamp_batch)

            loss_batch.backward() 
            optimizer_train.step() 
            
            # Append to epoch
            train_loss_batch_list = torch.cat((train_loss_batch_list, loss_batch.detach().reshape(-1))) 
        
        # Do val loop
        ds_model_net.eval() 
        with torch.no_grad():     
        
            for pdbid_batch, _, val_variant_batch, val_x_ds_batch, val_vamp_batch in dataloader_val:

                # check that all examples are from same pdb (so they all would have the same std curve)
                assert np.all(pdbid_batch[0] == np.array(pdbid_batch))

                # Compute predictions
                val_vamp_pred_batch = ds_model_net(val_x_ds_batch)

                # Transform predictions using standard curve
                val_vamp_std_pred_batch = standard_curve_network_dict[pdbid_batch[0][0]](val_vamp_pred_batch)

                # Compute loss
                val_loss_batch = loss_func(
                    val_vamp_std_pred_batch, val_vamp_batch
                )

                # Append to epoch
                val_loss_batch_list = torch.cat(
                    (val_loss_batch_list, val_loss_batch.reshape(-1))
                )
                
                val_vamp = torch.cat((val_vamp, val_vamp_batch.reshape(-1)), 0)

                val_vamp_pred = torch.cat(
                    (val_vamp_pred, val_vamp_pred_batch.reshape(-1)), 0
                )

                val_vamp_std_pred = torch.cat(
                    (val_vamp_std_pred, val_vamp_std_pred_batch.reshape(-1)), 0
                )

                val_variant.append(np.concatenate(val_variant_batch))  
        
        # Compute epoch metrics
        train_loss_list.append(train_loss_batch_list.mean().cpu().item())
        val_loss_list.append(val_loss_batch_list.mean().cpu().item())
        val_vamp = val_vamp.detach().cpu()    
        val_vamp_pred = val_vamp_pred.detach().cpu()
        val_vamp_std_pred = val_vamp_std_pred.detach().cpu()

        pearson_r_list.append(pearsonr(val_vamp_std_pred.numpy(), val_vamp.numpy())[0])

        val_vamp_list.append(val_vamp.numpy())
        val_vamp_pred_list.append(val_vamp_pred.numpy())
        val_vamp_std_pred_list.append(val_vamp_std_pred.numpy())
        val_variant_list.append(np.concatenate(val_variant))
        
        # Save from downstream model from current epoch
        torch.save(ds_model_net.state_dict(),f"{outpath}/ds_model_{model_idx}/{outdir}/model_{epoch}.pt")
        
        # Save standard curve models from current epoch
        for pdbid in pdbid_list:
            torch.save(
                standard_curve_network_dict[pdbid].state_dict(),
                f"{outpath}/ds_model_{model_idx}/{outdir}/standard_curve_{pdbid}_{epoch}.pt"
            )
        
        # Compare loss to current best loss and modify patience
        loss_val = val_loss_batch_list.mean().cpu().item()  
        if loss_val < current_best_loss_val:
            current_best_loss_val = loss_val
            current_best_epoch_idx = epoch
            patience = 0
        else:
            patience += 1
            
        # Early stopping
        if patience == PATIENCE:
            break
            
    # Check that val_variant_list is the same in every epoch
    for i in range(epoch):
        assert np.all(val_variant_list[0] == val_variant_list[i])
    
    # Save results evaluated over all epochs
    lc_results = (pearson_r_list, val_loss_list, train_loss_list, val_vamp_list,
                  val_vamp_pred_list, val_vamp_std_pred_list, val_variant_list, 
                  pdbid_val, model_idx)
    with open(f"{outpath}/ds_model_{model_idx}/{outdir}/lc_results.pkl", "wb") as f:
        pickle.dump(lc_results, f)
    
    # Copy downstream model from best epoch
    shutil.copyfile(f"{outpath}/ds_model_{model_idx}/{outdir}/model_{current_best_epoch_idx}.pt",
                    f"{outpath}/ds_model_{model_idx}/{outdir}/model.pt")
    
    # Copy standard curve models from best epoch 
    for pdbid in pdbid_list:
        shutil.copyfile(f"{outpath}/ds_model_{model_idx}/{outdir}/standard_curve_{pdbid}_{current_best_epoch_idx}.pt",
                        f"{outpath}/ds_model_{model_idx}/{outdir}/standard_curve_{pdbid}.pt")
    
    # Delete saved ds_model_net files that are not from best epoch
    ds_model_net_files = glob.glob(f'{outpath}/ds_model_{model_idx}/{outdir}/model_*')
    for f in ds_model_net_files:
        os.remove(f)
    
    # Delete standard curve network files that are not from best epoch
    for pdbid in pdbid_list:
        standard_curve_net_files = glob.glob(f'{outpath}/ds_model_{model_idx}/{outdir}/standard_curve_{pdbid}_*')
        for f in standard_curve_net_files:
            os.remove(f)

    # Print
    print(
        f"Best epoch: {current_best_epoch_idx + 1} with validation loss: "
        f"{current_best_loss_val:5.3f}"
    )    
    
    
def ds_std_train_val(
    dataloader_train,
    dataloader_train_val_std,
    dataloader_val,
    ds_model_input_size,
    use_pretrained_ds_model,
    std_train,
    loss_func,
    LEARNING_RATE_DS, 
    pdbid_val,
    model_idx,
    EPOCHS,
    PATIENCE,
    DEVICE,
    hidden_size_1,
    hidden_size_2
):
    """
    Function to train and validate downstream/abundance network. 
    Use when training with trainable standard curve. 
    """
    # Set seed
    np.random.seed(model_idx)
    random.seed(model_idx)
    torch.manual_seed(model_idx)
    torch.cuda.manual_seed(model_idx)
    torch.cuda.manual_seed_all(model_idx) 
    
    # Define pdbids
    pdbid_list = ['P60484','P51580','P11712','Q9NV35','P45381','O60260']
    pdbid_train_list = pdbid_list.copy()
    pdbid_train_list.remove(pdbid_val)
    
    # Load pretrained downstream model or intitialise model with random weights
    if use_pretrained_ds_model == True:
        
        if hidden_size_2 > 0:
            ds_model_net = DownstreamModel2(ds_model_input_size, hidden_size_1, hidden_size_2).to(DEVICE)
        else:
            ds_model_net = DownstreamModel1(ds_model_input_size, hidden_size_1).to(DEVICE)
        
        ds_model_net.load_state_dict(
            torch.load(f"../output/ds_models/leave_out_{pdbid_val}/ds_model_{model_idx}/std_fixed/model.pt",
            map_location=torch.device(DEVICE))    
        )   
        
    else:
        
        if hidden_size_2 > 0:
            ds_model_net = DownstreamModel2(ds_model_input_size, hidden_size_1, hidden_size_2).to(DEVICE)
        else:
            ds_model_net = DownstreamModel1(ds_model_input_size, hidden_size_1).to(DEVICE)
            
        torch.manual_seed(seed=model_idx)
        torch.cuda.manual_seed(seed=model_idx)
        ds_model_net.apply(init_lin_weights)
    
    # Initialise standard curve networks
    standard_curve_network_dict = {}
    for pdbid in pdbid_list:
        standard_curve_network_dict[pdbid] = init_standard_curve_network(pdbid, DEVICE)
    
    # Define optimizers depending on whether standard curve nets should be trained or not
    if std_train == True:
        # Define parameters to be learned during training
        params_train = list(ds_model_net.parameters())
        for pdbid in pdbid_train_list:
            params_train += list(standard_curve_network_dict[pdbid].parameters())
        # Define optimizers for (i) ds_net and training protein std curves
        # and (ii) validation protein std curve
        optimizer_train = torch.optim.Adam(params_train, lr=LEARNING_RATE_DS)
        optimizer_val = torch.optim.Adam(standard_curve_network_dict[pdbid_val].parameters(), lr=LEARNING_RATE_DS) 
    else:
        # Define optimiser for ds_net training
        optimizer_train = torch.optim.Adam(ds_model_net.parameters(), lr=LEARNING_RATE_DS)
    
    # Define parameters for early stopping
    current_best_epoch_idx = -1
    current_best_loss_val = 1e4
    patience = 0
    
    # Define name of output directory    
    if std_train == True:
        outdir = "std_train"
    else:
        outdir = "std_fixed"
    outpath = f"../output/ds_models/leave_out_{pdbid_val}"
    
    # Create output directory
    if not os.path.isdir(outpath):
            os.mkdir(outpath)
    if not os.path.isdir(f'{outpath}/ds_model_{model_idx}'):
            os.mkdir(f'{outpath}/ds_model_{model_idx}')
    if not os.path.isdir(f'{outpath}/ds_model_{model_idx}/{outdir}'):
            os.mkdir(f'{outpath}/ds_model_{model_idx}/{outdir}')
    
    # Initialize lists
    train_loss_list = []
    val_loss_list = []
    pearson_r_list = []

    val_vamp_list = []
    val_vamp_pred_list = []
    val_vamp_std_pred_list = []
    val_variant_list = []
    
    train_val_std_loss_list = []
    train_val_std_variant_list = []
    
    # Run training and validation over EPOCHS
    for epoch in range(EPOCHS):

        print(f"Epoch: {epoch+1}/{EPOCHS}")

        # Initialize
        train_loss_batch_list = torch.empty(0, dtype=torch.float32).to(DEVICE)
        val_loss_batch_list = torch.empty(0, dtype=torch.float32).to(DEVICE)
        val_vamp = torch.empty(0, dtype=torch.float32).to(DEVICE)
        val_vamp_pred = torch.empty(0, dtype=torch.float32).to(DEVICE)
        val_vamp_std_pred = torch.empty(0, dtype=torch.float32).to(DEVICE)
        val_vamp = torch.empty(0, dtype=torch.float32).to(DEVICE)
        val_variant = []
        train_val_std_loss_batch_list = torch.empty(0, dtype=torch.float32).to(DEVICE)
        train_val_std_variant = []
        
        # Train loop
        ds_model_net.train()  

        for pdbid_batch, _, _, x_ds_batch, vamp_batch in dataloader_train:
            
            # Initialize optimizer
            optimizer_train.zero_grad()

            # Compute predictions 
            vamp_batch_pred = ds_model_net(x_ds_batch)
            
            # Transform data with standard curve specific to each protein
            vamp_std_batch_pred = torch.empty(len(pdbid_batch), dtype=torch.float32, requires_grad=True).to(DEVICE)
            for b in range(len(pdbid_batch)): 
                vamp_std_batch_pred[b] = standard_curve_network_dict[pdbid_batch[b][0]](vamp_batch_pred[b])            
            
            # calculate loss
            loss_batch = loss_func(vamp_std_batch_pred.reshape(-1,1), vamp_batch)

            loss_batch.backward() 
            optimizer_train.step() 
            
            # Append to epoch
            train_loss_batch_list = torch.cat((train_loss_batch_list, loss_batch.detach().reshape(-1))) 
        
        # Do train val loop 
        ds_model_net.eval() 
        
        for pdbid_batch, _, val_variant_batch, val_x_ds_batch, val_vamp_batch in dataloader_train_val_std:
            
            # check that all examples are from same pdb (so they all would have the same std curve)
            assert np.all(pdbid_batch[0] == np.array(pdbid_batch))
            
            if std_train == True:
                # Initialize optimizer
                optimizer_val.zero_grad() 
                                        
            # Compute predictions
            val_vamp_pred_batch = ds_model_net(val_x_ds_batch)

            # Transform predictions using standard curve
            val_vamp_std_pred_batch = standard_curve_network_dict[pdbid_batch[0][0]](val_vamp_pred_batch) 
    
            # Calculate loss
            train_val_loss_batch = loss_func(val_vamp_std_pred_batch,val_vamp_batch)

            if std_train == True:
                train_val_loss_batch.backward()
                optimizer_val.step()

            # Append to epoch 
            train_val_std_loss_batch_list = torch.cat((train_val_std_loss_batch_list, 
                                                       train_val_loss_batch.detach().reshape(-1)))         
            train_val_std_variant.append(np.concatenate(val_variant_batch))
        
        # Do val loop
        with torch.no_grad():     
        
            for (pdbid_batch,
                 _,
                 val_variant_batch,
                 val_x_ds_batch,
                 val_vamp_batch) in dataloader_val:

                # check that all examples are from same pdb (so they all would have the same std curve)
                assert np.all(pdbid_batch[0] == np.array(pdbid_batch))

                # Compute predictions
                val_vamp_pred_batch = ds_model_net(val_x_ds_batch)

                # Transform predictions using standard curve
                val_vamp_std_pred_batch = standard_curve_network_dict[pdbid_batch[0][0]](val_vamp_pred_batch)

                # Compute loss
                val_loss_batch = loss_func(
                    val_vamp_std_pred_batch, val_vamp_batch
                )

                # Append to epoch
                val_loss_batch_list = torch.cat(
                    (val_loss_batch_list, val_loss_batch.reshape(-1))
                )
                
                val_vamp = torch.cat((val_vamp, val_vamp_batch.reshape(-1)), 0)

                val_vamp_pred = torch.cat(
                    (val_vamp_pred, val_vamp_pred_batch.reshape(-1)), 0
                )

                val_vamp_std_pred = torch.cat(
                    (val_vamp_std_pred, val_vamp_std_pred_batch.reshape(-1)), 0
                )

                val_variant.append(np.concatenate(val_variant_batch))  
        
        # Compute epoch metrics
        train_loss_list.append(train_loss_batch_list.mean().cpu().item())
        val_loss_list.append(val_loss_batch_list.mean().cpu().item())
        val_vamp = val_vamp.detach().cpu()    
        val_vamp_pred = val_vamp_pred.detach().cpu()
        val_vamp_std_pred = val_vamp_std_pred.detach().cpu()

        pearson_r_list.append(pearsonr(val_vamp_std_pred.numpy(), val_vamp.numpy())[0])

        val_vamp_list.append(val_vamp.numpy())
        val_vamp_pred_list.append(val_vamp_pred.numpy())
        val_vamp_std_pred_list.append(val_vamp_std_pred.numpy())
        val_variant_list.append(np.concatenate(val_variant))
        
        train_val_std_loss_list.append(train_val_std_loss_batch_list.mean().cpu().item())
        train_val_std_variant_list.append(np.concatenate(train_val_std_variant))
        
        # Save from downstream model from current epoch
        torch.save(ds_model_net.state_dict(),f"{outpath}/ds_model_{model_idx}/{outdir}/model_{epoch}.pt")
        
        # Save standard curve models from current epoch
        for pdbid in pdbid_list:
            torch.save(
                standard_curve_network_dict[pdbid].state_dict(),
                f"{outpath}/ds_model_{model_idx}/{outdir}/standard_curve_{pdbid}_{epoch}.pt"
            )
        
        # Compare loss to current best loss and modify patience
        loss_val = val_loss_batch_list.mean().cpu().item()  
        if loss_val < current_best_loss_val:
            current_best_loss_val = loss_val
            current_best_epoch_idx = epoch
            patience = 0
        else:
            patience += 1
            
        # Early stopping
        if patience == PATIENCE:
            break
            
    # Check that no variants were used for both training of val prot std curve and actual validation 
    assert len(np.intersect1d(np.concatenate(val_variant_list),np.concatenate(train_val_std_variant_list))) == 0 
    
    # Check that val_variant_list is the same in every epoch
    for i in range(epoch):
        assert np.all(val_variant_list[0] == val_variant_list[i])
    
    # Check that no variants in train_val_std_variant_list occur in val_variant_list
    for i in range(epoch):
        assert len(np.intersect1d(val_variant_list[i],train_val_std_variant_list[i])) == 0

    # Save results evaluated over all epochs
    lc_results = (pearson_r_list, val_loss_list, train_loss_list, val_vamp_list,
                  val_vamp_pred_list, val_vamp_std_pred_list, val_variant_list, 
                  pdbid_val, model_idx, train_val_std_loss_list, train_val_std_variant_list)
    with open(f"{outpath}/ds_model_{model_idx}/{outdir}/lc_results.pkl", "wb") as f:
        pickle.dump(lc_results, f)
    
    # Copy downstream model from best epoch
    shutil.copyfile(f"{outpath}/ds_model_{model_idx}/{outdir}/model_{current_best_epoch_idx}.pt",
                    f"{outpath}/ds_model_{model_idx}/{outdir}/model.pt")
    
    # Copy standard curve models from best epoch (if they have been trained)
    if std_train == True:
        for pdbid in pdbid_list:
            shutil.copyfile(f"{outpath}/ds_model_{model_idx}/{outdir}/standard_curve_{pdbid}_{current_best_epoch_idx}.pt",
                            f"{outpath}/ds_model_{model_idx}/{outdir}/standard_curve_{pdbid}.pt")
    
    # Delete saved ds_model_net files that are not from best epoch
    ds_model_net_files = glob.glob(f'{outpath}/ds_model_{model_idx}/{outdir}/model_*')
    for f in ds_model_net_files:
        os.remove(f)
    
    # Delete standard curve network files that are not from best epoch
    for pdbid in pdbid_list:
        standard_curve_net_files = glob.glob(f'{outpath}/ds_model_{model_idx}/{outdir}/standard_curve_{pdbid}_*')
        for f in standard_curve_net_files:
            os.remove(f)

    # Print
    print(
        f"Best epoch: {current_best_epoch_idx + 1} with validation loss: "
        f"{current_best_loss_val:5.3f}"
    )


def ds_pred(
    pdbid_val,
    dataloader_val,
    ds_model_input_size,
    std_type,
    model_idx,
    DEVICE,
    hidden_size_1,
    hidden_size_2
):
    """
    Run trained abundance network to predict low-throughput
    scores for variants of input protein. 
    """  
    # Set seed
    np.random.seed(model_idx)
    random.seed(model_idx)
    torch.manual_seed(model_idx)
    torch.cuda.manual_seed(model_idx)
    torch.cuda.manual_seed_all(model_idx)
    
    # Define name of output directory   
    outpath = f"../output/ds_models/leave_out_{pdbid_val}"
    outdir = std_type
    
    # Load trained model
    if hidden_size_2 > 0:
        ds_model_net = DownstreamModel2(ds_model_input_size, hidden_size_1, hidden_size_2).to(DEVICE)
    else:
        ds_model_net = DownstreamModel1(ds_model_input_size, hidden_size_1).to(DEVICE)
    ds_model_net.load_state_dict(
        torch.load(f"../output/ds_models/leave_out_{pdbid_val}/ds_model_{model_idx}/{outdir}/model.pt",
        map_location=torch.device(DEVICE))    
    )
    
    # Initialize arrays
    val_vamp = torch.empty(0, dtype=torch.float32).to(DEVICE)
    val_vamp_pred = torch.empty(0, dtype=torch.float32).to(DEVICE)
    val_variant_list = []
    
    # Evaluate all variants in validation protein
    ds_model_net.eval() 
    with torch.no_grad():
        
        for pdbid_batch, _, val_variant_batch, val_x_ds_batch, val_vamp_batch in dataloader_val:
            
            # Compute downstream network predictions
            val_vamp_pred_batch = ds_model_net(val_x_ds_batch)
            
            # Append 
            val_vamp = torch.cat((val_vamp, val_vamp_batch.reshape(-1)), 0)
            val_vamp_pred = torch.cat((val_vamp_pred, val_vamp_pred_batch.reshape(-1)), 0)
            val_variant_list.append(np.concatenate(val_variant_batch))
            
    val_vamp_list = val_vamp.detach().cpu()    
    val_vamp_pred_list = val_vamp_pred.detach().cpu()    
    
    val_variant_list = np.concatenate(val_variant_list)
    
    # Save results 
    val_results = (val_vamp_list, val_vamp_pred_list, val_variant_list, pdbid_val, model_idx)
    with open(f"{outpath}/ds_model_{model_idx}/{outdir}/val_results.pkl", "wb") as f:
        pickle.dump(val_results, f)        
        
        
def ds_std_pred(
    pdbid_val,
    dataloader_val,
    ds_model_input_size,
    std_type,
    model_idx,
    DEVICE,
    hidden_size_1,
    hidden_size_2
):
    """
    Run trained abundance network to predict low-throughput
    scores for variants of input protein, and pass predictions
    through standard curves. 
    """  
    # Set seed
    np.random.seed(model_idx)
    random.seed(model_idx)
    torch.manual_seed(model_idx)
    torch.cuda.manual_seed(model_idx)
    torch.cuda.manual_seed_all(model_idx)
    
    # Define name of output directory   
    outpath = f"../output/ds_models/leave_out_{pdbid_val}"
    outdir = std_type
    
    # Load trained model
    if hidden_size_2 > 0:
        ds_model_net = DownstreamModel2(ds_model_input_size, hidden_size_1, hidden_size_2).to(DEVICE)
    else:
        ds_model_net = DownstreamModel1(ds_model_input_size, hidden_size_1).to(DEVICE)
    ds_model_net.load_state_dict(
        torch.load(f"../output/ds_models/leave_out_{pdbid_val}/ds_model_{model_idx}/{outdir}/model.pt",
        map_location=torch.device(DEVICE))    
    )
    
    # Load standard curve
    standard_curve_network = init_standard_curve_network(pdbid_val, DEVICE)
    
    # Initialize arrays
    val_vamp = torch.empty(0, dtype=torch.float32).to(DEVICE)
    val_vamp_pred = torch.empty(0, dtype=torch.float32).to(DEVICE)
    val_vamp_std_pred = torch.empty(0, dtype=torch.float32).to(DEVICE)
    val_variant_list = []
    
    # Evaluate all variants in validation protein
    ds_model_net.eval() 
    with torch.no_grad():
        
        for (pdbid_batch, _, val_variant_batch, val_x_ds_batch, val_vamp_batch) in dataloader_val:
            
            # Compute downstream network predictions
            val_vamp_pred_batch = ds_model_net(val_x_ds_batch)
            
            # Run predictions through standard curve
            val_vamp_std_pred_batch = standard_curve_network(val_vamp_pred_batch)
            
            # Append 
            val_vamp = torch.cat((val_vamp, val_vamp_batch.reshape(-1)), 0)
            val_vamp_pred = torch.cat((val_vamp_pred, val_vamp_pred_batch.reshape(-1)), 0)
            val_vamp_std_pred = torch.cat((val_vamp_std_pred, val_vamp_std_pred_batch.reshape(-1)), 0)
            val_variant_list.append(np.concatenate(val_variant_batch))
            
    val_vamp_list = val_vamp.detach().cpu()    
    val_vamp_pred_list = val_vamp_pred.detach().cpu()   
    val_vamp_std_pred_list = val_vamp_std_pred.detach().cpu()   
    
    val_variant_list = np.concatenate(val_variant_list)
    
    # Save results 
    val_results = (val_vamp_list, val_vamp_pred_list, val_vamp_std_pred_list, val_variant_list, pdbid_val, model_idx)
    with open(f"{outpath}/ds_model_{model_idx}/{outdir}/val_results.pkl", "wb") as f:
        pickle.dump(val_results, f)
