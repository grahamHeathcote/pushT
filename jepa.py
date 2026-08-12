from neural_network import Network

class Jepa():
    def __init__(self, latent_dims, enc1, enc2, pred1):
        self.encoder = Network(12, enc1, enc2, latent_dims)
        self.predictor = Network(latent_dims, pred1, pred1, latent_dims)

        