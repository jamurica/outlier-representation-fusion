"""Train a CPU autoencoder and extract its latent representation."""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .representations import reduced_dimension


class Autoencoder(nn.Module):
    """A symmetric fully connected autoencoder with a linear bottleneck."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        latent_dim = reduced_dimension(n_features)
        hidden_dim = max(16, 2 * latent_dim)
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim, device="cpu"),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim, device="cpu"),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim, device="cpu"),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_features, device="cpu"),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(features))

    @torch.inference_mode()
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Encode standardized rows in their original order."""
        inputs = torch.as_tensor(features, dtype=torch.float32, device="cpu")
        return self.encoder(inputs).numpy()


def fit_autoencoder(train: np.ndarray, *, seed: int) -> Autoencoder:
    """Fit the autoencoder on standardized training features."""
    if train.ndim != 2 or train.size == 0 or not np.isfinite(train).all():
        raise ValueError("Expected a nonempty, finite training feature matrix")

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = Autoencoder(train.shape[1]).float()

    inputs = torch.as_tensor(train, dtype=torch.float32, device="cpu")
    batches = DataLoader(
        inputs, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.MSELoss()
    for _ in range(50):
        for batch in batches:
            optimizer.zero_grad()
            loss = loss_function(model(batch), batch)
            loss.backward()
            optimizer.step()
    return model.eval()
