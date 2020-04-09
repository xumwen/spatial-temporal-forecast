import os
import time
import argparse
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import TestTubeLogger

from pytorch_lightning.callbacks import EarlyStopping

from argparse import Namespace

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from stgcn import STGCN
from tgcn import TGCN
from gwnet import GWNET
from preprocess import generate_dataset, load_nyc_sharing_bike_data, load_metr_la_data, get_normalized_adj


parser = argparse.ArgumentParser(description='Spatial-Temporal-Model')
parser.add_argument('--backend', choices=['dp', 'ddp'],
                    help='Backend for data parallel', default='ddp')
parser.add_argument('--log-name', type=str, default='default',
                    help='Experiment name to log')
parser.add_argument('--log-dir', type=str, default='./logs',
                    help='Path to log dir')
parser.add_argument('--gpus', type=int, default=1,
                    help='Number of GPUs to use')
parser.add_argument('-m', "--model", choices=['tgcn', 'stgcn', 'gwnet'],
                    help='Choose Spatial-Temporal model', default='stgcn')
parser.add_argument('-d', "--dataset", choices=["metr", "nyc-bike"],
                    help='Choose dataset', default='metr')
parser.add_argument('-t', "--gcn-type", 
                    choices=['normal', 'cheb', 'sage', 'graph', 'gat', 'egnn', 'sagela'],
                    help='Choose GCN Conv Type', default='cheb')
parser.add_argument('-p', "--gcn-package", choices=['pyg', 'ours'],
                    help='Choose GCN implemented package',
                    default='ours')
parser.add_argument('-part', "--gcn-partition", choices=['cluster', 'sample'],
                    help='Choose GCN partition method',
                    default=None)
parser.add_argument('-batchsize', type=int, default=64,
                    help='Training batch size')
parser.add_argument('-epochs', type=int, default=1000,
                    help='Training epochs')
parser.add_argument('-l', '--loss-criterion', choices=['mse', 'mae'],
                    help='Choose loss criterion', default='mse')
parser.add_argument('-num-timesteps-input', type=int, default=12,
                    help='Num of input timesteps')
parser.add_argument('-num-timesteps-output', type=int, default=3,
                    help='Num of output timesteps for forecasting')
parser.add_argument('-early-stop-rounds', type=int, default=10,
                    help='Earlystop rounds when validation loss does not decrease')

args = parser.parse_args()
if torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')
    
model = {'tgcn':TGCN, 'stgcn':STGCN, 'gwnet':GWNET}.get(args.model)
backend = args.backend
log_name = args.log_name
log_dir = args.log_dir
gpus = args.gpus

loss_criterion = {'mse': nn.MSELoss(), 'mae': nn.L1Loss()}\
    .get(args.loss_criterion)
gcn_type = args.gcn_type
gcn_package = args.gcn_package
gcn_partition = args.gcn_partition
batch_size = args.batchsize
epochs = args.epochs
num_timesteps_input = args.num_timesteps_input
num_timesteps_output = args.num_timesteps_output
early_stop_rounds = args.early_stop_rounds


class WrapperNet(pl.LightningModule):
    # NOTE: pl module is supposed to only have ``hparams`` parameter
    def __init__(self, hparams):
        super(WrapperNet, self).__init__()

        self.hparams = hparams
        self.net = model(
            hparams.num_nodes,
            hparams.num_edges,
            hparams.num_features,
            hparams.num_timesteps_input,
            hparams.num_timesteps_output,
            hparams.gcn_type,
            hparams.gcn_package,
            hparams.gcn_partition
        )
        self.register_buffer('A', torch.Tensor(
            hparams.num_nodes, hparams.num_nodes).float())
        self.register_buffer('edge_index', torch.LongTensor(
            2, hparams.num_edges))
        self.register_buffer('edge_weight', torch.Tensor(
            hparams.num_edges).float())

    def init_graph(self, A, edge_index, edge_weight):
        self.A.copy_(A)
        self.edge_index.copy_(edge_index)
        self.edge_weight.copy_(edge_weight)

    def init_data(self, training_input, training_target, val_input, val_target, test_input, test_target):
        print('preparing data...')
        self.training_input = training_input
        self.training_target = training_target
        self.val_input = val_input
        self.val_target = val_target
        self.test_input = test_input
        self.test_target = test_target

    def make_dataloader(self, X, y, shuffle, backend=backend):
        dataset = TensorDataset(X, y)

        if backend == 'dp':
            return DataLoader(dataset, batch_size=batch_size, num_workers=1, shuffle=shuffle, drop_last=True)
        elif backend == 'ddp':
            dist_sampler = DistributedSampler(dataset, shuffle=shuffle)
            return DataLoader(dataset, batch_size=batch_size, num_workers=0, sampler=dist_sampler)

    def train_dataloader(self):
        return self.make_dataloader(self.training_input, self.training_target, shuffle=True)

    def val_dataloader(self):
        return [
            self.make_dataloader(
                self.val_input, self.val_target, shuffle=False),
            self.make_dataloader(
                self.test_input, self.test_target, shuffle=False),
        ]

    def test_dataloader(self):
        return self.make_dataloader(self.test_input, self.test_target, shuffle=False, backend='dp')

    def forward(self, X):
        return self.net(X, A=self.A, 
                        edge_index=self.edge_index, 
                        edge_weight=self.edge_weight)

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        assert(y.size() == y_hat.size())
        loss = loss_criterion(y_hat, y)

        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx, dataloader_idx):
        X, y = batch
        y_hat = self(X)
        return {'loss': loss_criterion(y_hat, y)}

    def validation_end(self, outputs):
        tqdm_dict = dict()
        for idx, output in enumerate(outputs):
            prefix = 'val' if idx == 0 else 'test'
            loss_mean = torch.stack([x['loss'] for x in output]).mean()
            tqdm_dict[prefix + '_loss'] = loss_mean
        self.logger.experiment.flush()
        return {'progress_bar': tqdm_dict, 'log': tqdm_dict}

    def test_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        return {'loss': loss_criterion(y_hat, y)}

    def test_end(self, outputs):
        loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        print('Mean test loss : {}'.format(loss_mean.item()))
        tqdm_dict = {'test_loss': loss_mean}
        return {'progress_bar': tqdm_dict, 'log': tqdm_dict}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


if __name__ == '__main__':
    start_time = time.time()
    print('cuda available:', torch.cuda.is_available())
    print("device:", args.device)
    print("model:", args.model)
    print("dataset:", args.dataset)
    print("gcn type:", args.gcn_type)
    print("gcn package:", args.gcn_package)
    print("gcn partition:", args.gcn_partition)
    torch.manual_seed(7)

    if args.dataset == "metr":
        A, X, means, stds = load_metr_la_data()
    else:
        A, X, means, stds = load_nyc_sharing_bike_data()

    split_line1 = int(X.shape[2] * 0.6)
    split_line2 = int(X.shape[2] * 0.8)

    train_original_data = X[:, :, :split_line1]
    val_original_data = X[:, :, split_line1:split_line2]
    test_original_data = X[:, :, split_line2:]

    training_input, training_target = generate_dataset(train_original_data,
                                                       num_timesteps_input=num_timesteps_input,
                                                       num_timesteps_output=num_timesteps_output)
    val_input, val_target = generate_dataset(val_original_data,
                                             num_timesteps_input=num_timesteps_input,
                                             num_timesteps_output=num_timesteps_output)
    test_input, test_target = generate_dataset(test_original_data,
                                               num_timesteps_input=num_timesteps_input,
                                               num_timesteps_output=num_timesteps_output)

    A = torch.from_numpy(A)
    sparse_A = A.to_sparse()
    edge_index = sparse_A._indices()
    edge_weight = sparse_A._values()

    hparams = Namespace(**{
        'num_nodes': A.shape[0],
        'num_edges': edge_weight.shape[0],
        'num_features': training_input.shape[3],
        'num_timesteps_input': num_timesteps_input,
        'num_timesteps_output': num_timesteps_output,
        'gcn_type': gcn_type,
        'gcn_package': gcn_package,
        'gcn_partition': gcn_partition
    })

    net = WrapperNet(hparams)

    net.init_data(
        training_input, training_target,
        val_input, val_target,
        test_input, test_target
    )

    net.init_graph(A, edge_index, edge_weight)

    early_stop_callback = EarlyStopping(patience=early_stop_rounds)
    logger = TestTubeLogger(save_dir=log_dir, name=log_name)

    trainer = pl.Trainer(
        gpus=[1],
        max_epochs=epochs,
        distributed_backend=backend,
        early_stop_callback=early_stop_callback,
        logger=logger,
        track_grad_norm=2
    )
    trainer.fit(net)

    print('Training time {}'.format(time.time() - start_time))

    # # Currently, there are some issues for testing under ddp setting, so switch it to dp setting
    # # change the below line with your own checkpoint path
    # net = WrapperNet.load_from_checkpoint('logs/ddp_exp/version_1/checkpoints/_ckpt_epoch_2.ckpt')
    # net.init_data(
    #     training_input, training_target,
    #     val_input, val_target,
    #     test_input, test_target
    # )
    # tester = pl.Trainer(
    #     gpus=gpus,
    #     max_epochs=epochs,
    #     distributed_backend='dp',
    # )
    # tester.test(net)
