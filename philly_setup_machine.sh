#! /bin/bash

# Option 1: use system python and install necessary packages on the fly

# CUDA=cu101
# yes | pip install --user torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html
# yes | pip install --user torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html
# yes | pip install --user torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html
# yes | pip install --user torch-cluster
# yes | pip install --user torch-geometric
# yes | pip install tensorboardX

# Option 2: create your own conda and env in advance (recommended because installing pyg is time-consuming)
# Note: you need to specify proper env when doing setup for each process

# You can install new packages here (done once for each container)
# Already installed packages:

# CONDA_HOME="/var/storage/shared/resrchvc/shunzhen/anaconda3"
# export CONDA_ENVS_PATH=$CONDA_HOME/envs
# export CONDA_ENVS_DIR=$CONDA_HOME/envs
# export CONDA_PKGS_PATH=$CONDA_HOME/envs
# export CONDA_PKGS_DIR=$CONDA_HOME/envs
# export PATH="$CONDA_HOME/bin:$PATH"
# source activate gnn
# CUDA=cu101
# yes | pip install tqdm
# yes | pip install torch
# yes | pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html
# yes | pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html
# yes | pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html
# yes | pip install torch-cluster
# yes | pip install torch-geometric
# yes | pip install tensorboardX
# source deactivate