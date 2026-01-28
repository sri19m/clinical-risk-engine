import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    """
    Compresses high-dimensional clinical data into latent embeddings.
    """
    def __init__(self, input_dim: int, latent_dim: int = 8):
        super(Autoencoder, self).__init__()
        # Encoder: Compresses input
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        # Decoder: Reconstructs input
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

class MultiTaskNet(nn.Module):
    """
    Shared-representation network for predicting Stroke, Lung Cancer, 
    and Phenotype Cluster simultaneously.
    """
    def __init__(self, input_dim: int, shared_dim: int = 64):
        super(MultiTaskNet, self).__init__()
        
        # Shared Layers (Feature Extractor)
        self.shared = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(shared_dim, shared_dim // 2),
            nn.ReLU(),
        )
        
        # Task-Specific Heads
        self.stroke_head = nn.Sequential(
            nn.Linear(shared_dim // 2, 32), 
            nn.ReLU(), 
            nn.Linear(32, 1) # Binary classification
        )
        self.lung_head = nn.Sequential(
            nn.Linear(shared_dim // 2, 32), 
            nn.ReLU(), 
            nn.Linear(32, 1) # Binary classification
        )
        self.pheno_head = nn.Sequential(
            nn.Linear(shared_dim // 2, 32), 
            nn.ReLU(), 
            nn.Linear(32, 3) # Multi-class classification (3 clusters)
        )

    def forward(self, x):
        h = self.shared(x)
        s_logit = self.stroke_head(h).squeeze(1)
        l_logit = self.lung_head(h).squeeze(1)
        p_logit = self.pheno_head(h)
        return s_logit, l_logit, p_logit