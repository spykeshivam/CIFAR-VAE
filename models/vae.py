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


class ResidualBlock(nn.Module):
    """
    Residual block for the VAE.
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection if dimensions change
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        residual = x
        
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = F.leaky_relu(out, 0.2)
        
        return out


class ConvVAE(nn.Module):
    """
    Improved Convolutional Variational Autoencoder (ConvVAE) for image data.
    
    Args:
        in_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
        img_size (int): Size of the input images (assuming square images)
        hidden_dims (list): List of hidden dimensions for the convolutional layers
        latent_dim (int): Dimensionality of the latent space
        beta (float): Weight for the KL divergence term in the loss function
    """
    
    def __init__(self, in_channels=3, img_size=64, hidden_dims=[32, 64, 128, 256], latent_dim=128, beta=1.0):
        super(ConvVAE, self).__init__()
        
        self.in_channels = in_channels
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Build Encoder
        modules = []
        
        # Initial convolution
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dims[0], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(hidden_dims[0]),
                nn.LeakyReLU(0.2)
            )
        )
        
        # Downsampling convolutions
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    ResidualBlock(hidden_dims[i], hidden_dims[i]),
                    nn.Conv2d(hidden_dims[i], hidden_dims[i+1], kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.LeakyReLU(0.2)
                )
            )
        
        # Add one more residual block at the deepest level
        modules.append(ResidualBlock(hidden_dims[-1], hidden_dims[-1]))
        
        self.encoder = nn.Sequential(*modules)
        
        # Calculate the size of the feature maps before flattening
        self.feature_size = img_size // (2 ** (len(hidden_dims) - 1))
        self.flatten_size = hidden_dims[-1] * self.feature_size * self.feature_size
        
        # Latent space projections with variance clamping
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        # Build Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flatten_size)
        
        modules = []
        
        # Reshape layer
        self.reshape_layer = hidden_dims[-1]
        
        # Decoder layers
        hidden_dims.reverse()
        
        # First residual block after reshaping
        self.initial_decoder_block = ResidualBlock(hidden_dims[0], hidden_dims[0])
        
        # Transposed convolutions for upsampling
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                      hidden_dims[i + 1],
                                      kernel_size=4,
                                      stride=2,
                                      padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(0.2),
                    ResidualBlock(hidden_dims[i + 1], hidden_dims[i + 1])
                )
            )
        
        # Final layer with sigmoid activation
        modules.append(
            nn.Sequential(
                nn.Conv2d(hidden_dims[-1], self.in_channels, kernel_size=3, padding=1),
                nn.Sigmoid()
            )
        )
        
        self.decoder = nn.Sequential(*modules)
    
    def encode(self, x):
        """
        Encodes the input images into the latent space.
        
        Args:
            x (torch.Tensor): Input images of shape [batch_size, channels, height, width]
        
        Returns:
            tuple: (mu, logvar) parameters of the latent distribution
        """
        # Encode
        x = self.encoder(x)
        
        # Flatten
        x = torch.flatten(x, start_dim=1)
        
        # Get latent parameters
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterizes the encoded data to sample from the latent space. Done to allow backpropagation. Differentiable gradient.
        
        Args:
            mu (torch.Tensor): Mean of the latent distribution
            logvar (torch.Tensor): Log variance of the latent distribution
        
        Returns:
            torch.Tensor: Sampled data from the latent space
        """
        # Clamp logvar to prevent numerical issues
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
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
        
        # Initial residual block
        z = self.initial_decoder_block(z)
        
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
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        
        # Create a distribution (needed for the VAEOutput dataclass)
        scale = torch.exp(0.5 * logvar)
        scale_tril = torch.diag_embed(scale)
        dist = torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril)
        
        if not compute_loss:
            return VAEOutput(
                z_dist=dist,
                z_sample=z,
                x_recon=recon_x
            )
        
        # Compute loss terms
        if self.in_channels == 1:  # Grayscale images
            loss_recon = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
        else:  # Color images
            # Use a combination of BCE and MSE for better reconstructions
            bce_loss = F.binary_cross_entropy(recon_x, x, reduction='sum') / x.size(0)
            mse_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
            loss_recon = 0.8 * bce_loss + 0.2 * mse_loss
        
        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        
        # Apply beta weighting to KL term (beta-VAE)
        loss = loss_recon + self.beta * loss_kl
        
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
    
    def reconstruct(self, x):
        """
        Reconstructs the input images.
        
        Args:
            x (torch.Tensor): Input images of shape [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Reconstructed images of shape [batch_size, channels, height, width]
        """
        with torch.no_grad():
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            recon_x = self.decode(z)
        return recon_x