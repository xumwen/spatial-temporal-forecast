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
from base_task import add_config_to_argparse, BaseConfig, BasePytorchTask, \
    LOSS_KEY, BAR_KEY, SCALAR_LOG_KEY, VAL_SCORE_KEY


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
                    choices=['normal', 'cheb', 'sage', 'graph', 'gat'],
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
parser.add_argument('-early-stop-rounds', type=int, default=30,
                    help='Earlystop rounds when validation loss does not decrease')
# TODO:
# 1. define SpatialTemporalConfig
# 2. use the following function to add arguments
# 3. refactor the configuration setup accordingly
add_config_to_argparse(BaseConfig(), parser)

args = parser.parse_args()
args_dict = dict(args.__dict__)
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


class WrapperNet(nn.Module):
    def __init__(self, hparams):
        super().__init__()

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

    def forward(self, X):
        return self.net(X, A=self.A,
                        edge_index=self.edge_index,
                        edge_weight=self.edge_weight)


class SpatialTemporalTask(BasePytorchTask):
    def __init__(self, config):
        super(SpatialTemporalTask, self).__init__(config)

    def init_data(self, training_input, training_target, val_input, val_target, test_input, test_target):
        self.log('preparing data...')
        self.training_input = training_input
        self.training_target = training_target
        self.val_input = val_input
        self.val_target = val_target
        self.test_input = test_input
        self.test_target = test_target

    def make_dataloader(self, X, y, shuffle=False, use_distributed=False):
        dataset = TensorDataset(X, y)

        if use_distributed:
            dist_sampler = DistributedSampler(dataset, shuffle=shuffle)
            # the total batch size will be batch_size x process_num x gradient_accumulation_steps
            return DataLoader(dataset, batch_size=batch_size, sampler=dist_sampler)
        else:
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def build_train_dataloader(self):
        return self.make_dataloader(
            self.training_input, self.training_target,
            shuffle=True, use_distributed=self.in_distributed_mode
        )

    def build_val_dataloader(self):
        return self.make_dataloader(self.val_input, self.val_target)

    def build_test_dataloader(self):
        return self.make_dataloader(self.test_input, self.test_target)

    def build_optimizer(self, model):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def train_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.model(X)
        assert(y.size() == y_hat.size())
        loss = loss_criterion(y_hat, y)
        loss_i = loss.item()  # scalar loss

        return {
            LOSS_KEY: loss,
            BAR_KEY: { 'train_loss': loss_i },
            SCALAR_LOG_KEY: { 'train_loss': loss_i }
        }

    def eval_step(self, batch, batch_idx, tag):
        X, y = batch
        y_hat = self.model(X)
        loss = loss_criterion(y_hat, y)
        return {
            LOSS_KEY: loss,
            BAR_KEY: { '{}_loss'.format(tag): loss.item() },
        }

    def eval_epoch_end(self, outputs, tag):
        loss = torch.stack(
            [x[LOSS_KEY] for x in outputs]
        ).mean().item()
        out = {
            BAR_KEY: { '{}_loss'.format(tag) : loss },
            SCALAR_LOG_KEY: { '{}_loss'.format(tag) : loss },
            VAL_SCORE_KEY: -loss,
        }
        # self.log(out[BAR_KEY])
        return out

    def val_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'val')

    def val_epoch_end(self, outputs):
        return self.eval_epoch_end(outputs, 'val')

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'test')

    def test_epoch_end(self, outputs):
        return self.eval_epoch_end(outputs, 'test')


if __name__ == '__main__':
    start_time = time.time()
    # build config and task for Spatial-Temporal Forecasting
    st_config = BaseConfig()
    # TODO: simplify the argument parsing
    st_config.update_by_dict(args_dict)
    st_task = SpatialTemporalTask(st_config)

    st_task.log("cuda available: {}".format(torch.cuda.is_available()))
    st_task.log("device: {}".format(args.device))
    st_task.log("model: {}".format(args.model))
    st_task.log("dataset: {}".format(args.dataset))
    st_task.log("gcn type: {}".format(args.gcn_type))
    st_task.log("gcn package: {}".format(args.gcn_package))
    st_task.log("gcn partition: {}".format(args.gcn_partition))
    torch.manual_seed(7)

    # TODO: Integrate the data processing pipeline into st_task.__init__()
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

    st_task.init_data(
        training_input, training_target,
        val_input, val_target,
        test_input, test_target
    )

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

    # Set random seed before the initialization of network parameters
    # Necessary for ddp training
    st_task.set_random_seed()
    net = WrapperNet(hparams)

    net.init_graph(A, edge_index, edge_weight)

    # TODO: add support for early stopping
    if st_task.config.skip_train:
        st_task.init_model_and_optimizer(net)
    else:
        st_task.fit(net)

    # Resume the best checkpoint for evaluation
    st_task.resume_best_checkpoint()
    val_eval_out = st_task.val_eval()
    test_eval_out = st_task.test_eval()
    st_task.log('Best checkpoint (epoch={}, {}, {})'.format(
        st_task._passed_epoch, val_eval_out[BAR_KEY], test_eval_out[BAR_KEY]))

    st_task.log('Training time {}'.format(time.time() - start_time))
