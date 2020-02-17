import os
import argparse
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from stgcn import STGCN
from tgcn import TGCN
from preprocess import generate_dataset, load_metr_la_data, get_normalized_adj


parser = argparse.ArgumentParser(description='Spatial-Temporal-Model')
parser.add_argument('--enable-cuda', action='store_true',
                    help='Enable CUDA')
parser.add_argument('-m', "--model", choices=['tgcn', 'stgcn'], 
		    help='Choice Spatial-Temporal model', default='tgcn')
parser.add_argument('-batch_size', type=int, default=64,
		    help='Training batch size')
parser.add_argument('-epochs', type=int, default=1000,
		    help='Training epochs')
parser.add_argument('-num_timesteps_input', type=int, default=15,
		    help='Num of input timesteps')
parser.add_argument('-num_timesteps_output', type=int, default=3,
		    help='Num of output timesteps for forecasting')

args = parser.parse_args()
if args.enable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')
if args.model == 'tgcn':
    model = TGCN
else:
    model = STGCN
batch_size = args.batch_size
epochs = args.epochs
num_timesteps_input = args.num_timesteps_input
num_timesteps_output = args.num_timesteps_output

def train_epoch(training_input, training_target, batch_size):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """
    permutation = torch.randperm(training_input.shape[0])

    epoch_training_losses = []
    for i in range(0, training_input.shape[0], batch_size):
        net.train()
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        X_batch, y_batch = training_input[indices], training_target[indices]
        X_batch = X_batch.to(device=args.device)
        y_batch = y_batch.to(device=args.device)
        
        out = net(A_wave, X_batch)
        loss = loss_criterion(out, y_batch)
        if i / batch_size % 100 == 0:
            print('After training %d batches, loss = %lf' % (i / batch_size, loss.item()))
        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())
    return sum(epoch_training_losses)/len(epoch_training_losses)


if __name__ == '__main__':
    print('cuda available:', torch.cuda.is_available())
    print("device:", args.device)
    print("model:", args.model)
    torch.manual_seed(11)

    A, X, means, stds = load_metr_la_data()

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

    A_wave = torch.from_numpy(get_normalized_adj(A)).to(device=args.device)

    net = model(A_wave.shape[0],
                training_input.shape[3],
                num_timesteps_input,
                num_timesteps_output).to(device=args.device)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_criterion = nn.MSELoss()

    training_losses = []
    validation_losses = []
    validation_maes = []
    for epoch in range(epochs):
        print('=' * 50, 'epoch %d'%(epoch+1), '=' * 50)
        loss = train_epoch(training_input, training_target,
                           batch_size=batch_size)
        training_losses.append(loss)

        # Run validation
        with torch.no_grad():
            net.eval()
            val_input = val_input.to(device=args.device)
            val_target = val_target.to(device=args.device)

            out = net(A_wave, val_input)
            val_loss = loss_criterion(out, val_target).to(device="cpu")
            validation_losses.append(np.asscalar(val_loss.detach().numpy()))

            out_unnormalized = out.detach().cpu().numpy()*stds[0]+means[0]
            target_unnormalized = val_target.detach().cpu().numpy()*stds[0]+means[0]
            mae = np.mean(np.absolute(out_unnormalized - target_unnormalized))
            validation_maes.append(mae)

            out = None
            val_input = val_input.to(device="cpu")
            val_target = val_target.to(device="cpu")

        print("Training loss: {}".format(training_losses[-1]))
        print("Validation loss: {}".format(validation_losses[-1]))
        print("Validation MAE: {}".format(validation_maes[-1]))
        plt.plot(training_losses, label="training loss")
        plt.plot(validation_losses, label="validation loss")
        plt.legend()
        plt.show()

        checkpoint_path = "checkpoints/"
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        with open("checkpoints/losses.pk", "wb") as fd:
            pk.dump((training_losses, validation_losses, validation_maes), fd)
