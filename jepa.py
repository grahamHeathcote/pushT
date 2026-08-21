from neural_network import Network
from torch import nn
class Jepa(nn.Module):
    def __init__(self, latent_dims, hidden):
        super().__init__()
        self.encoder = Network(9, hidden, hidden, latent_dims)
        self.predictor = Network(latent_dims + 2, hidden, hidden, latent_dims)