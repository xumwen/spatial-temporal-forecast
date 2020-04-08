# Spatial-Temporal-Forecast
Some baseline model of Spatial-Temporal Forecasting.

 ## Models
 
  * T-GCN

PyTorch implementation of the spatio-temporal graph convolutional network proposed in [T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction](https://arxiv.org/abs/1811.05320) by Ling Zhao, Yujiao Song, Chao Zhang, Yu Liu, Pu Wang, Tao Lin, Min Deng, Haifeng Li. 

  * STGCN

Pytorch version of stgcn proposed in [Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting](https://arxiv.org/abs/1709.04875) by Bing Yu, Haoteng Yin, Zhanxing Zhu.
This version was implemented by [FelixOpolka](https://github.com/FelixOpolka/STGCN-PyTorch).

  * Graph Wavenet

Pytorch version of graph wavenet proposed in [Graph WaveNet for Deep Spatial-Temporal Graph Modeling](https://arxiv.org/abs/1906.00121) by Zonghan Wu, Shirui Pan, Guodong Long, Jing Jiang, Chengqi Zhang.
This version was implemented by [nnzhan](https://github.com/nnzhan/Graph-WaveNet).
And there is a little refactoring on it. 
  
 ## Requirements
  
  * PyTorch
  * NumPy
  * Pytorch_geometric
  * Pytorch_lightning
  * TestTube
  
 ## Example Dataset
  
  * METR-LA

The repository provides a usage example on the METR-LA dataset (original version to be found [here](https://github.com/liyaguang/DCRNN)).

  * NYC Sharing Bike

Origin source is [citibikenyc](https://www.citibikenyc.com/system-data).
Example data is [201307-201402-citibike-tripdata.zip](https://s3.amazonaws.com/tripdata/index.html)

 ## Commands
```
python main.py -m gwnet -d metr
python main.py -m stgcn -d nyc-bike -t cheb -p pyg
```

 ## Use BasePytorchTask to substitute pytorch-lightning
```
# Single-GPU Training Example: Use GPU:1
CUDA_VISIBLE_DEVICES=1 python main_task.py -m stgcn

# Single-Machine Distributed Training Example: Use GPU:0 and GPU:3 with 2 Processes
CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node 2 main_task.py -m stgcn
```
