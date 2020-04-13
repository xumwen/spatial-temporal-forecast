#! /bin/bash

# Setup conda env for each process
CONDA_HOME="/var/storage/shared/resrchvc/shunzhen/anaconda3"
export CONDA_ENVS_PATH=$CONDA_HOME/envs
export CONDA_ENVS_DIR=$CONDA_HOME/envs
export CONDA_PKGS_PATH=$CONDA_HOME/envs
export CONDA_PKGS_DIR=$CONDA_HOME/envs
export PATH="$CONDA_HOME/bin:$PATH"
source activate gnn