import torch
import torch.nn as nn
from torch.distributions import Independent
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal
from utils.utils import get_device
from utils import utils


def IWAE_reconstruction_loss(batch_dict, encoder, decoder, z0_prior, obsrv_std,
                             n_traj_samples=1, run_backwards=True, kl_coef=1.):
    truth = batch_dict["observed_data"]  # [batch_size, n_tp, n_feature]
    mask = batch_dict["observed_mask"]  # [batch_size, n_tp, n_feature]
    truth_time_steps = batch_dict["observed_tp"]  # [n_tp]
    truth_to_predict = batch_dict["data_to_predict"]  # [batch_size, n_tp_pred, n_feature]
    mask_predict = batch_dict["mask_predicted_data"]  # [batch_size, n_tp_pred, n_feature]
    time_steps_to_predict = batch_dict["tp_to_predict"]  # [n_tp_pred]

    # encoder
    # truth_w_mask = torch.cat((truth, mask), dim=-1) if mask is not None else truth
    first_point_mu, first_point_std = encoder(truth, mask, truth_time_steps, run_backwards=run_backwards)
    assert (torch.sum(first_point_std < 0) == 0.)

    # Reparameterization
    means_z0 = first_point_mu.repeat(n_traj_samples, 1, 1)
    sigma_z0 = first_point_std.repeat(n_traj_samples, 1, 1)
    first_point_enc = utils.sample_standard_gaussian(means_z0, sigma_z0)

    # decoder
    sol_z, pred_x = decoder(first_point_enc, time_steps_to_predict)

    # IWAE loss
    kl_z0 = kl_divergence(Normal(first_point_mu, first_point_std), z0_prior)
    kl_z0 = torch.mean(kl_z0, (1, 2))
    rec_likelihood = compute_masked_likelihood(pred_x=pred_x,
                                               truth=truth_to_predict.repeat(n_traj_samples, 1, 1, 1),
                                               mask=mask_predict.repeat(n_traj_samples, 1, 1, 1),
                                               func=lambda pred, truth_p: Gaussian_likelihood(pred, truth_p, obsrv_std))
    rec_likelihood = torch.mean(rec_likelihood, 1)
    loss = - torch.logsumexp(rec_likelihood - kl_coef * kl_z0, dim=0)
    if torch.isnan(loss):
        loss = - torch.mean(rec_likelihood - kl_coef * kl_z0, dim=0)
    return first_point_enc, sol_z, pred_x, loss, rec_likelihood, kl_z0, first_point_std


def compute_binary_CE_loss(labels_pred, labels):
    """
    Compute the binary cross entropy loss for binary classification problems.
    :param labels_pred: [n_traj_samples, batch_size]
    :param labels: [batch_size, 1]
    :return: [1]
    """
    labels = labels.reshape(-1)
    n_traj_samples = labels_pred.size(0)

    if torch.sum(labels == 0.) == 0 or torch.sum(labels == 1.) == 0:
        print("Warning: all examples in a batch belong to the same class -- please increase the batch size.")

    labels = labels.repeat(n_traj_samples, 1)
    ce_loss = nn.BCEWithLogitsLoss()(labels_pred, labels)
    return ce_loss


def Gaussian_likelihood(pred_x, truth, obsrv_std):
    n_data_points = pred_x.size()[-1]

    if n_data_points > 0:
        gaussian = Independent(Normal(loc=pred_x, scale=obsrv_std.repeat(n_data_points)), 1)
        log_prob = gaussian.log_prob(truth)
        log_prob = log_prob / n_data_points
    else:
        log_prob = torch.zeros([1]).to(get_device(truth)).squeeze()
    return log_prob


def mse(pred_x, truth):
    n_data_points = pred_x.size()[-1]
    mse_loss = nn.MSELoss()(pred_x, truth) if n_data_points > 0 \
        else torch.zeros([1]).to(get_device(truth)).squeeze()
    return mse_loss


def compute_masked_likelihood(pred_x, truth, mask, func):
    """
    Compute the masked func likelihood of truth given the pred_x.
    :param pred_x: [n_traj_samples, batch_size, n_tp, n_feature]
    :param truth: [n_traj_samples, batch_size, n_tp, n_feature]
    :param mask: [n_traj_samples, batch_size, n_tp, n_feature]
    :param func: likelihood function
    :return: likelihood: [n_traj_samples, batch_size]
    """
    n_traj_samples, batch_size, n_tp, n_feature = pred_x.size()

    if mask is None:
        pred_x_flat = pred_x.reshape(n_traj_samples * batch_size, n_tp * n_feature)
        truth_flat = truth.reshape(n_traj_samples * batch_size, n_tp * n_feature)
        result = func(pred_x_flat, truth_flat)
        if result.reshape(-1).size(0) == n_traj_samples * batch_size:
            result = result.reshape(n_traj_samples, batch_size)
    else:
        # Compute the likelihood per patient and per attribute so that we don't priorize patients with more measurements
        result = []
        for i in range(n_traj_samples):
            for j in range(batch_size):
                for k in range(n_feature):
                    pred_x_mask = torch.masked_select(pred_x[i, j, :, k], mask[i, j, :, k].bool())
                    truth_mask = torch.masked_select(truth[i, j, :, k], mask[i, j, :, k].bool())
                    likelihood = func(pred_x_mask, truth_mask)
                    result.append(likelihood)
        result = torch.stack(result, 0).to(get_device(truth))
        result = result.reshape(n_traj_samples, batch_size, n_feature)
        result = torch.mean(result, -1)

    return result
