import torch
import torch.nn as nn

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1,1,40,130)
            out = self.encoder(dummy)
            self.flatten_dim = out.shape[1]

        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        self.fc = nn.Linear(latent_dim, self.flatten_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.flatten_dim, 64*10*33),
            nn.ReLU(),
            nn.Unflatten(1, (64,10,33)),
            nn.ConvTranspose2d(64,32,3,2,1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32,1,3,2,1,output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)

        x = self.fc(z)
        x = self.decoder(x)

        return x, mu, logvar, z