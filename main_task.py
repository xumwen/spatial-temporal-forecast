import os
import time
import argparse
import json
from argparse import Namespace
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt

from stgcn import STGCN
from tgcn import TGCN
from gwnet import GWNET
from preprocess import generate_dataset, load_nyc_sharing_bike_data, load_metr_la_data, get_normalized_adj
from base_task import add_config_to_argparse, BaseConfig, BasePytorchTask, \
    LOSS_KEY, BAR_KEY, SCALAR_LOG_KEY, VAL_SCORE_KEY


class STConfig(BaseConfig):
    def __init__(self):
        super(STConfig, self).__init__()
        self.model = 'stgcn'  # choices: tgcn, stgcn, gwnet
        self.dataset = 'metr'  # choices: metr, nyc
        self.data_dir = './data/METR-LA'  # choices: ./data/METR-LA, ./data/NYC-Sharing-Bike
        self.gcn = 'cheb'  # choices: normal, cheb, sage, graph, gat
        self.gcn_package = 'ours'  # choices: pyg, ours
        self.gcn_partition = 'none'  # choices: none, cluster, sample
        self.batch_size = 64  # per-gpu training batch size, real_batch_size = batch_size * num_gpus * grad_accum_steps
        self.loss = 'mse'  # choices: mse, mae
        self.num_timesteps_input = 12  # the length of the input time-series sequence
        self.num_timesteps_output = 3  # the length of the output time-series sequence
        self.lr = 1e-3  # the learning rate


def get_model_class(model):
    return {
        'tgcn': TGCN,
        'stgcn': STGCN,
        'gwnet': GWNET,
    }.get(model)


def get_loss_func(loss):
    return {
        'mse': nn.MSELoss(),
        'mae': nn.L1Loss(),
    }.get(loss)


class WrapperNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        model_class = get_model_class(config.model)
        gcn_partition = None if config.gcn_partition is 'none' else config.gcn_partition
        self.net = model_class(
            config.num_nodes,
            config.num_edges,
            config.num_features,
            config.num_timesteps_input,
            config.num_timesteps_output,
            config.gcn,
            config.gcn_package,
            gcn_partition
        )
        self.register_buffer('A', torch.Tensor(
            config.num_nodes, config.num_nodes).float())
        self.register_buffer('edge_index', torch.LongTensor(
            2, config.num_edges))
        self.register_buffer('edge_weight', torch.Tensor(
            config.num_edges).float())

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
        self.log('Intialize {}'.format(self.__class__))

        self.init_data()
        self.loss_func = get_loss_func(config.loss)

        self.config.num_nodes = self.A.shape[0]
        self.config.num_edges = self.edge_weight.shape[0]
        self.config.num_features = self.training_input.shape[3]

        self.log('Config:\n{}'.format(
            json.dumps(self.config.to_dict(), ensure_ascii=False, indent=4)
        ))


    def init_data(self, data_dir=None):
        if data_dir is None:
            data_dir = self.config.data_dir

        if self.config.dataset == "metr":
            A, X, means, stds = load_metr_la_data(data_dir)
        else:
            A, X, means, stds = load_nyc_sharing_bike_data(data_dir)

        split_line1 = int(X.shape[2] * 0.6)
        split_line2 = int(X.shape[2] * 0.8)
        train_original_data = X[:, :, :split_line1]
        val_original_data = X[:, :, split_line1:split_line2]
        test_original_data = X[:, :, split_line2:]

        self.training_input, self.training_target = generate_dataset(train_original_data,
            num_timesteps_input=self.config.num_timesteps_input, num_timesteps_output=self.config.num_timesteps_output
        )
        self.val_input, self.val_target = generate_dataset(val_original_data,
            num_timesteps_input=self.config.num_timesteps_input, num_timesteps_output=self.config.num_timesteps_output
        )
        self.test_input, self.test_target = generate_dataset(test_original_data,
            num_timesteps_input=self.config.num_timesteps_input, num_timesteps_output=self.config.num_timesteps_output
        )

        self.A = torch.from_numpy(A)
        self.sparse_A = self.A.to_sparse()
        self.edge_index = self.sparse_A._indices()
        self.edge_weight = self.sparse_A._values()

    def make_dataloader(self, X, y, shuffle=False, use_distributed=False):
        dataset = TensorDataset(X, y)

        if use_distributed:
            dist_sampler = DistributedSampler(dataset, shuffle=shuffle)
            # the total batch size will be batch_size x process_num x gradient_accumulation_steps
            return DataLoader(dataset, batch_size=self.config.batch_size, sampler=dist_sampler)
        else:
            return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=shuffle)

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
        return torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

    def train_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.model(X)
        assert(y.size() == y_hat.size())
        loss = self.loss_func(y_hat, y)
        loss_i = loss.item()  # scalar loss

        return {
            LOSS_KEY: loss,
            BAR_KEY: { 'train_loss': loss_i },
            SCALAR_LOG_KEY: { 'train_loss': loss_i }
        }

    def eval_step(self, batch, batch_idx, tag):
        X, y = batch
        y_hat = self.model(X)
        loss = self.loss_func(y_hat, y)
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

    # build argument parser and config
    st_config = STConfig()
    parser = argparse.ArgumentParser(description='Spatial-Temporal-Task')
    add_config_to_argparse(st_config, parser)

    # parse arguments to config
    args = parser.parse_args()
    st_config.update_by_dict(args.__dict__)

    # build task
    st_task = SpatialTemporalTask(st_config)

    # Set random seed before the initialization of network parameters
    # Necessary for distributed training
    st_task.set_random_seed()
    net = WrapperNet(st_task.config)
    net.init_graph(st_task.A, st_task.edge_index, st_task.edge_weight)

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

    st_task.log('Training time {}s'.format(time.time() - start_time))
