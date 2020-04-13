#! /bin/bash

# Option 1: use system python and install necessary packages on the fly

# CUDA=cu101
# yes | pip install --user torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html
# yes | pip install --user torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html
# yes | pip install --user torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html
# yes | pip install --user torch-cluster
# yes | pip install --user torch-geometric
# yes | pip install tensorboardX
# yes | pip install tqdm

# Option 2: create your own conda env on the shared NFS in advance (recommended, because installing pyg is time-consuming)
# You should not install new packages here because this script is executed by each machine
# You should not switch conda env here because this option should be done for each process

# You can do machine-level preparations here
sudo apt-get update