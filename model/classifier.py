import torch.nn as nn
from utils import utils


class Classifier(nn.Module):
    def __init__(self, latent_dim, n_labels, ues_linear_classifier=False):
        super(Classifier, self).__init__()
        if ues_linear_classifier:
            self.classifier = nn.Sequential(nn.Linear(latent_dim, n_labels))
        else:
            self.classifier = nn.Sequential(
                nn.Linear(latent_dim, 300),
                nn.ReLU(),
                nn.Linear(300, 300),
                nn.ReLU(),
                nn.Linear(300, n_labels)
            )
        utils.init_network_weights(self.classifier)

    def forward(self, x):
        return self.classifier(x)
