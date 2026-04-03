import numpy as np
from sklearn.decomposition import PCA

class VAEExtractor:
    """
    Skeleton class for Variational Autoencoder (VAE) feature extraction.
    Currently acts as a functional placeholder using PCA.
    """
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

    def fit(self, data):
        """
        Fit the model on EEG data.
        :param data: EEG data (channels x samples)
        """
        # Input to sklearn algos as samples x features
        self.pca.fit(data.T)

    def transform(self, data):
        """
        Extract latent representation.
        :param data: EEG data (channels x samples)
        :return: Latent representation (n_components x samples)
        """
        return self.pca.transform(data.T).T

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
