import torch
import torch.nn as nn
from model.loss import IWAE_reconstruction_loss, compute_binary_CE_loss, compute_masked_likelihood, mse, \
    compute_multiclass_CE_loss
from utils.utils import get_device


class VAE(nn.Module):
    def __init__(self, encoder, decoder, z0_prior, obsrv_std, classifier=None, classif_per_tp=False,
                 train_classif_w_reconstr=False):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.z0_prior = z0_prior
        self.obsrv_std = obsrv_std
        self.classifier = classifier
        self.classif_per_tp = classif_per_tp
        self.train_classif_w_reconstr = train_classif_w_reconstr

    def compute_all_losses(self, batch_dict, n_traj_samples=1, kl_coef=1.):
        first_point_enc, sol_z, pred_x, loss, rec_likelihood, kl_z0, first_point_std = (
            IWAE_reconstruction_loss(batch_dict, self.encoder, self.decoder, self.z0_prior, self.obsrv_std,
                                     n_traj_samples=n_traj_samples, kl_coef=kl_coef))

        mean_se = compute_masked_likelihood(pred_x=pred_x,
                                            truth=batch_dict['data_to_predict'].repeat(n_traj_samples, 1, 1, 1),
                                            mask=batch_dict['mask_predicted_data'].repeat(n_traj_samples, 1, 1, 1),
                                            func=mse)
        pois_log_likelihood = torch.Tensor([0.]).to(get_device(batch_dict["data_to_predict"]))

        device = get_device(batch_dict["data_to_predict"])
        ce_loss = torch.Tensor([0.]).to(device)
        if self.classifier is not None and batch_dict['labels'] is not None:
            if self.classif_per_tp:
                labels_prediction = self.classifier(sol_z)
            else:
                labels_prediction = self.classifier(first_point_enc).squeeze(-1)

            if (batch_dict["labels"].size(-1) == 1) or (len(batch_dict["labels"].size()) == 1):
                ce_loss = compute_binary_CE_loss(labels_prediction, batch_dict['labels'])
            else:
                ce_loss = compute_multiclass_CE_loss(
                    labels_prediction,
                    batch_dict['labels'],
                    mask=batch_dict['mask_predicted_data']
                )

            if self.train_classif_w_reconstr:
                loss = loss + ce_loss * 30
            else:
                loss = ce_loss

        results = {
            'loss': torch.mean(loss),
            'likelihood': torch.mean(rec_likelihood).detach(),
            'mse': torch.mean(mean_se).detach(),
            'pois_log_likelihood': torch.mean(pois_log_likelihood).detach(),
            'ce_loss': torch.mean(ce_loss).detach(),
            'kl_first_p': torch.mean(kl_z0).detach(),
            'std_first_p': torch.mean(first_point_std).detach(),
        }

        if self.classifier is not None and batch_dict['labels'] is not None:
            results['label_predictions'] = labels_prediction.detach()

        return results
