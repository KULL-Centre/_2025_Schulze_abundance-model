# _2025_Schulze_abundance-model

This repository contains data and scripts to reproduce results from the following preprint:

Thea K. Schulze, Lasse M. Blaabjerg, Matteo Cagiada, Kresten Lindorff-Larsen (2025) *Supervised learning of protein variant effects across large-scale mutagenesis datasets*. bioRxiv 2025.04.02.646878; doi: https://doi.org/10.1101/2025.04.02.646878

-----

We provide example scripts and data to train and validate a supervised model against VAMP-seq abundance scores:

`/output/models.zip` contains selected, trained models.
`/output/feature_set_1` contains examples of scripts to run training and validation pipeline for different versions of our model architecture. 
All training data and model input features can be found in `/data/df_vamp.csv`.
