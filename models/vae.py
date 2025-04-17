import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class VAEOutput:
    """
    Dataclass for VAE output.
    
    Attributes:
        z_dist (torch.distributions.Distribution): The distribution of the latent variable z.
        z_sample (torch.Tensor): The sampled value of the latent variable z.
        x_recon (torch.Tensor): The reconstructed output from the VAE.
        loss (torch.Tensor): The overall loss of the VAE.
        loss_recon (torch.Tensor): The reconstruction loss component of the VAE loss.
        loss_kl (torch.Tensor): The KL divergence component of the VAE loss.
    """
    z_dist: torch.distributions.Distribution
    z_sample: torch.Tensor
    x_recon: torch.Tensor
    
    loss: torch.Tensor = None
    loss_recon: torch.Tensor = None
    loss_kl: torch.Tensor = None


class ConvVAE(nn.Module):
    """
    Convolutional Variational Autoencoder (ConvVAE) for image data.
    
    Args:
        in_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
        img_size (int): Size of the input images (assuming square images)
        hidden_dims (list): List of hidden dimensions for the convolutional layers
        latent_dim (int): Dimensionality of the latent space
    """
    
    def __init__(self, in_channels=3, img_size=64, hidden_dims=[32, 64, 128, 256], latent_dim=128):
        super(ConvVAE, self).__init__()
        
        self.in_channels = in_channels
        self.img_size = img_size
        self.latent_dim = latent_dim
        
        # Build Encoder
        modules = []
        
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU()
                )
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        
        # Calculate the size of the feature maps before flattening
        self.feature_size = img_size // (2 ** len(hidden_dims))
        self.flatten_size = hidden_dims[-1] * self.feature_size * self.feature_size
        
        # Latent space projections
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_var = nn.Linear(self.flatten_size, latent_dim)
        self.softplus = nn.Softplus()
        
        # Build Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flatten_size)
        
        modules = []
        
        # Reshape layer
        self.reshape_layer = hidden_dims[-1]
        
        # Decoder layers
        hidden_dims.reverse()
        
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=4,
                                       stride=2,
                                       padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.SiLU()
                )
            )
        
        # Final layer
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1],
                                  self.in_channels,
                                  kernel_size=4,
                                  stride=2,
                                  padding=1),
                nn.Sigmoid()
            )
        )
        
        self.decoder = nn.Sequential(*modules)
    
    def encode(self, x, eps: float = 1e-8):
        """
        Encodes the input images into the latent space.
        
        Args:
            x (torch.Tensor): Input images of shape [batch_size, channels, height, width]
            eps (float): Small value to avoid numerical instability
        
        Returns:
            torch.distributions.MultivariateNormal: Normal distribution of the encoded data
        """
        # Encode
        x = self.encoder(x)
        
        # Flatten
        x = torch.flatten(x, start_dim=1)
        
        # Get latent parameters
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        
        # Create scale for distribution
        scale = self.softplus(logvar) + eps
        scale_tril = torch.diag_embed(scale)
        
        return torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril)
    
    def reparameterize(self, dist):
        """
        Reparameterizes the encoded data to sample from the latent space.
        
        Args:
            dist (torch.distributions.MultivariateNormal): Normal distribution of the encoded data
        
        Returns:
            torch.Tensor: Sampled data from the latent space
        """
        return dist.rsample()
    
    def decode(self, z):
        """
        Decodes the data from the latent space to reconstruct images.
        
        Args:
            z (torch.Tensor): Data in the latent space of shape [batch_size, latent_dim]
        
        Returns:
            torch.Tensor: Reconstructed images of shape [batch_size, channels, height, width]
        """
        # Project and reshape
        z = self.decoder_input(z)
        z = z.view(-1, self.reshape_layer, self.feature_size, self.feature_size)
        
        # Decode
        return self.decoder(z)
    
    def forward(self, x, compute_loss: bool = True):
        """
        Performs a forward pass of the ConvVAE.
        
        Args:
            x (torch.Tensor): Input images of shape [batch_size, channels, height, width]
            compute_loss (bool): Whether to compute the loss or not
        
        Returns:
            VAEOutput: VAE output dataclass
        """
        dist = self.encode(x)
        z = self.reparameterize(dist)
        recon_x = self.decode(z)
        
        if not compute_loss:
            return VAEOutput(
                z_dist=dist,
                z_sample=z,
                x_recon=recon_x
            )
        
        # Compute loss terms
        # For images, we use MSE or BCE loss for reconstruction
        if self.in_channels == 1:  # Grayscale images
            loss_recon = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
        else:  # Color images
            loss_recon = F.binary_cross_entropy(recon_x, x, reduction='sum') / x.size(0)
        
        # KL divergence
        std_normal = torch.distributions.MultivariateNormal(
            torch.zeros_like(z, device=z.device),
            scale_tril=torch.eye(z.shape[-1], device=z.device).unsqueeze(0).expand(z.shape[0], -1, -1),
        )
        loss_kl = torch.distributions.kl.kl_divergence(dist, std_normal).mean()
        
        # Total loss
        loss = loss_recon + loss_kl
        
        return VAEOutput(
            z_dist=dist,
            z_sample=z,
            x_recon=recon_x,
            loss=loss,
            loss_recon=loss_recon,
            loss_kl=loss_kl,
        )
    
    def sample(self, num_samples: int, device):
        """
        Generate samples from the latent space.
        
        Args:
            num_samples (int): Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            torch.Tensor: Generated samples [num_samples, channels, height, width]
        """
        # Sample from standard normal distribution
        z = torch.randn(num_samples, self.latent_dim, device=device)
        
        # Decode samples
        samples = self.decode(z)
        
        return samples