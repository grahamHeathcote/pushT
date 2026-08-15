from neural_network import Network
from torch import nn
class Jepa(nn.Module):
    def __init__(self, latent_dims, hidden):
        super().__init__()
        self.encoder = Network(12, hidden, hidden, latent_dims)
        self.decoder = Network(latent_dims, hidden, hidden, 5)
        self.predictor = Network(latent_dims + 2, hidden, hidden, latent_dims)