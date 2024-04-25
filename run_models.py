import os
import sys
import time
import datetime
import argparse
from random import SystemRandom

import numpy as np
import pandas as pd
from sklearn import model_selection

import torch
import torch.nn as nn
from torch.nn.functional import relu
import torch.optim as optim
from torch.distributions.normal import Normal

from model.vae import VAE
from model.decoder import ODEFunc, DiffeqSolver, DiffeqSolverDecoder
from model.encoder import Encoder_z0_ODE_RNN, EncoderAttention, Encoder_z0_RNN
from model.classifier import Classifier
from model.loss import IWAE_reconstruction_loss, compute_binary_CE_loss
from utils import utils
from utils.utils import compute_loss_all_batches
from parse_dataset import parse_datasets

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

parser = argparse.ArgumentParser('Latent ODE')

# model train hyperparameters
parser.add_argument('-n', type=int, default=100, help="Size of the dataset")
parser.add_argument('--niters', type=int, default=300)
parser.add_argument('--lr', type=float, default=1e-2, help="Starting learning rate.")
parser.add_argument('-b', '--batch-size', type=int, default=50)
parser.add_argument('-r', '--random-seed', type=int, default=1991, help="Random_seed")

# model save&load
parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None,
                    help="ID of the experiment to load for evaluation. If None, run a new experiment.")

# dataset selection
parser.add_argument('--dataset', type=str, default='periodic',
                    help="Dataset to load. Available: physionet, activity, hopper, periodic")
parser.add_argument('-s', '--sample-tp', type=float, default=None,
                    help="Number of time points to sub-sample. If > 1, subsample exact number of points. "
                         "If the number is in [0,1], take a percentage of available points per time series. "
                         "If None, do not subsample")
parser.add_argument('-c', '--cut-tp', type=int, default=None,
                    help="Cut out the section of the timeline of the specified length (in number of points)."
                         "Used for periodic function demo.")
parser.add_argument('--quantization', type=float, default=0.1,
                    help="Quantization on the physionet dataset."
                         "Value 1 means quantization by 1 hour, value 0.1 means quantization by 0.1 hour = 6 min")
parser.add_argument('-t', '--timepoints', type=int, default=100, help="Total number of time-points")
parser.add_argument('--max-t', type=float, default=5., help="We subsample points in the interval [0, args.max_tp]")
parser.add_argument('--noise-weight', type=float, default=0.01, help="Noise amplitude for generated trajectories")

# model selection
parser.add_argument('--latent-ode', action='store_true', help="Run Latent ODE seq2seq model")
parser.add_argument('--encoder', type=str, default='odernn',
                    help="Type of encoder for Latent ODE model: rnn , odernn, attn, mamba")
parser.add_argument('--classic-rnn', action='store_true',
                    help="Run RNN baseline: classic RNN that sees true points at every point. "
                         "Used for interpolation only.")
parser.add_argument('--rnn-cell', default="gru", help="RNN Cell type. Available: gru (default), exp decay")
parser.add_argument('--input-decay', action='store_true',
                    help="For RNN: use the input that is the weighted average of empirical mean and previous value "
                         "(like in GRU-D)")
parser.add_argument('--ode-rnn', action='store_true',
                    help="Run ODE-RNN baseline: RNN-style that sees true points at every point. "
                         "Used for interpolation only.")
parser.add_argument('--rnn-vae', action='store_true',
                    help="Run RNN baseline: seq2seq model with sampling of the h0 and ELBO loss.")

# model hyperparameters
parser.add_argument('-l', '--latents', type=int, default=6, help="Size of the latent state")
parser.add_argument('--rec-dims', type=int, default=20, help="Dimensionality of the recognition model (ODE or RNN).")
parser.add_argument('--rec-layers', type=int, default=1, help="Number of layers in ODE func in recognition ODE")
parser.add_argument('--gen-layers', type=int, default=1, help="Number of layers in ODE func in generative ODE")
parser.add_argument('-u', '--units', type=int, default=100, help="Number of units per layer in ODE func")
parser.add_argument('-g', '--gru-units', type=int, default=100,
                    help="Number of units per layer in each of GRU update networks")

# optional function
parser.add_argument('--poisson', action='store_true',
                    help="Model poisson-process likelihood for the density of events in addition to reconstruction.")
parser.add_argument('--classif', action='store_true',
                    help="Include binary classification loss -- used for Physionet dataset for hospital mortality")
parser.add_argument('--linear-classif', action='store_true',
                    help="If using a classifier, use a linear classifier instead of 1-layer NN")
parser.add_argument('--extrap', action='store_true',
                    help="Set extrapolation mode. If this flag is not set, run interpolation mode.")

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
file_name = os.path.basename(__file__)[:-3]
utils.makedirs(args.save)

if __name__ == '__main__':
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    experimentID = args.load if args.load is not None else int(SystemRandom().random() * 100000)
    ckpt_path = os.path.join(args.save, "experiment_" + str(experimentID) + '.ckpt')

    start = time.time()
    print("Sampling dataset of {} training examples".format(args.n))

    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind + 2):]
    input_command = ' '.join(input_command)

    utils.makedirs('results/')

    # dataset
    data_obj = parse_datasets(args, device)
    input_dim = data_obj['input_dim']
    classif_per_tp = data_obj['classif_per_tp'] if 'classif_per_tp' in data_obj else False

    n_labels = 1
    if args.classif:
        if 'n_labels' in data_obj:
            n_labels = data_obj['n_labels']
        else:
            raise Exception('Please provide number of labels for classification task')

    # model
    obsrv_std = 0.01
    if args.dataset == 'hopper':
        obsrv_std = 1e-3
    obsrv_std = torch.Tensor([obsrv_std]).to(device)
    z0_prior = Normal(torch.Tensor([0.]).to(device), torch.Tensor([1.]).to(device))

    if args.latent_ode:
        latents = args.latents
        gen_layers = args.gen_layers
        units = args.units
        rec_dims = args.rec_dims

        if args.encoder == 'rnn':
            encoder = Encoder_z0_RNN(latents, int(input_dim) * 2,
                                     lstm_output_size=rec_dims, device=device).to(device)
        elif args.encoder == 'odernn':
            rec_ode_func = ODEFunc(rec_dims, args.rec_layers, units, nonlinear=nn.Tanh).to(device)
            z0_diffeq_solver = DiffeqSolver(rec_ode_func, 'euler', odeint_rtol=1e-3, odeint_atol=1e-4).to(device)
            encoder = Encoder_z0_ODE_RNN(rec_dims, int(input_dim) * 2, z0_diffeq_solver,
                                         z0_dim=latents, n_gru_units=args.gru_units).to(device)
        elif args.encoder == 'attn':
            encoder = EncoderAttention(
                input_dim=input_dim,
                d_model=64,
                nhead=8,
                d_ff=512,
                num_layers=6,
                latent_dim=latents,
                dropout=0.5,
                use_split=False
            ).to(device)
        elif args.encoder == 'mamba':
            raise NotImplementedError('Not implemented yet')
        else:
            raise Exception('Please provide a valid encoder type')
        dec_ode_func = ODEFunc(latents, gen_layers, units, nonlinear=nn.Tanh).to(device)
        decoder = DiffeqSolverDecoder(latents, input_dim, dec_ode_func,
                                      method='dopri5', odeint_rtol=1e-3, odeint_atol=1e-4).to(device)
        classifier = Classifier(latents, n_labels, ues_linear_classifier=args.linear_classif).to(device) \
            if args.classif else None
        model = VAE(encoder=encoder,
                    decoder=decoder,
                    z0_prior=z0_prior,
                    obsrv_std=obsrv_std,
                    classifier=classifier,
                    classif_per_tp=classif_per_tp,
                    train_classif_w_reconstr=(args.dataset == "physionet")
                    ).to(device)
    else:
        raise Exception("Model not specified")

    # load checkpoint
    if args.load is not None:
        utils.get_ckpt_model(ckpt_path, model, device)
        exit()

    # training
    log_path = 'logs/' + file_name + '_' + str(experimentID) + '_' + args.encoder + '_' + args.dataset + '.log'
    utils.makedirs('logs/')
    logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
    logger.info(input_command)

    optimizer = optim.Adamax(model.parameters(), lr=args.lr)
    n_batches = data_obj['n_train_batches']

    for itr in range(1, n_batches * (args.niters + 1)):
        optimizer.zero_grad()
        utils.update_learning_rate(optimizer, decay_rate=0.999, lowest=args.lr / 10)

        wait_until_kl_inc = 10
        if itr // n_batches < wait_until_kl_inc:
            kl_coef = 0.
        else:
            kl_coef = (1 - 0.99 ** (itr // n_batches - wait_until_kl_inc))

        batch_dict = utils.get_next_batch(data_obj['train_dataloader'])
        train_res = model.compute_all_losses(batch_dict, n_traj_samples=3, kl_coef=kl_coef)
        train_res['loss'].backward()
        optimizer.step()

        n_iters_to_viz = 1
        if itr % (n_iters_to_viz * n_batches) == 0:
            with torch.no_grad():
                test_res = compute_loss_all_batches(
                    model=model,
                    test_dataloader=data_obj['test_dataloader'],
                    args=args,
                    n_batches=data_obj['n_test_batches'],
                    experimentID=experimentID,
                    device=device,
                    n_traj_samples=3,
                    kl_coef=kl_coef,
                )
                message = ('Epoch {:04d} [Test seq (cond on sampled tp)] | Loss {:.6f} | Likelihood {:.6f} | KL fp {'
                           ':.4f} | FP STD {:.4f} |').format(
                    itr // n_batches,
                    test_res['loss'].detach(),
                    test_res['likelihood'].detach(),
                    test_res['kl_first_p'].detach(),
                    test_res['std_first_p'].detach(),
                )

                logger.info('Experiment ' + str(experimentID))
                logger.info(message)
                logger.info('KL coef: {}'.format(kl_coef))
                logger.info('Train loss (one batch): {}'.format(train_res['loss'].detach()))
                logger.info('Train CE loss (one batch): {}'.format(test_res['ce_loss'].detach()))

                if 'auc' in test_res:
                    logger.info('Classification AUC (TEST): {:.4f}'.format(test_res['auc']))
                if 'mse' in test_res:
                    logger.info('TEST MSE: {:.4f}'.format(test_res['mse']))
                if 'accuracy' in train_res:
                    logger.info('Classification accuracy (TRAIN): {:.4f}'.format(train_res['accuracy']))
                if 'accuracy' in test_res:
                    logger.info('Classification accuracy (TEST): {:.4f}'.format(test_res['accuracy']))
                if 'ce_loss' in test_res:
                    logger.info('CE loss: {}'.format(test_res['ce_loss']))

            torch.save({
                'args': args,
                'state_dict': model.state_dict(),
            }, ckpt_path)

    torch.save({
        'args': args,
        'state_dict': model.state_dict(),
    }, ckpt_path)
