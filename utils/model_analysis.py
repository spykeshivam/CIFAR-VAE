import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from tqdm import tqdm
from models.vae import ConvVAE

def visualize_latent_space(model, dataloader, device, n_samples=1000):
    """
    Visualizes the latent space of the ConvVAE.
    
    Args:
        model (nn.Module): The trained ConvVAE model.
        dataloader (torch.utils.data.DataLoader): The data loader.
        device (torch.device): Device to run on.
        n_samples (int): Maximum number of samples to plot.
    """
    # Get the latent dimension from the model
    latent_dim = model.latent_dim
    
    if latent_dim != 2:
        print("Visualization only supported for 2D latent space.")
        return
    
    model.eval()
    z_samples = []
    labels = []
    sample_count = 0
    
    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="Collecting latent samples"):
            if sample_count >= n_samples:
                break
                
            data = data.to(device)
            output = model(data, compute_loss=False)
            z_samples.append(output.z_sample.cpu().numpy())
            labels.append(target.numpy())
            
            sample_count += len(data)
    
    z_samples = np.concatenate(z_samples, axis=0)[:n_samples]
    labels = np.concatenate(labels, axis=0)[:n_samples]
    
    # 2D scatter plot colored by class
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z_samples[:, 0], z_samples[:, 1], c=labels, cmap='tab10', 
                         alpha=0.5, s=5)
    plt.colorbar(scatter, label='Animal Class')
    plt.title('CIFAR-10 Animals 2D Latent Space')
    plt.xlabel('z[0]')
    plt.ylabel('z[1]')
    plt.savefig('convvae_cifar_latent_scatter.png')
    plt.close()
    
    # 2D histogram of the latent space
    plt.figure(figsize=(10, 8))
    plt.hist2d(z_samples[:, 0], z_samples[:, 1], bins=50, cmap='viridis', norm=plt.cm.colors.LogNorm())
    plt.colorbar(label='Count (log scale)')
    plt.title('CIFAR-10 Animals 2D Latent Space Density')
    plt.xlabel('z[0]')
    plt.ylabel('z[1]')
    plt.savefig('convvae_cifar_latent_hist.png')
    plt.close()
    
    # 1D marginals
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].hist(z_samples[:, 0], bins=50, alpha=0.7)
    axs[0].set_title('Marginal Distribution of z[0]')
    axs[0].set_xlabel('z[0]')
    axs[0].set_ylabel('Count')
    
    axs[1].hist(z_samples[:, 1], bins=50, alpha=0.7)
    axs[1].set_title('Marginal Distribution of z[1]')
    axs[1].set_xlabel('z[1]')
    axs[1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('convvae_cifar_latent_marginals.png')
    plt.close()


def interpolate_latent_space(model, dataloader, device, n_interp=10):
    """
    Creates interpolations in the latent space between real examples.
    
    Args:
        model (nn.Module): The trained ConvVAE model.
        dataloader (torch.utils.data.DataLoader): The data loader.
        device (torch.device): Device to run on.
        n_interp (int): Number of interpolation steps.
    """
    model.eval()
    
    # Get two real examples to interpolate between
    with torch.no_grad():
        # Get batch of real images
        data_batch, _ = next(iter(dataloader))
        data_batch = data_batch.to(device)
        
        # Get their latent representations
        output = model(data_batch[:2], compute_loss=False)
        z_start = output.z_sample[0:1]  # First example
        z_end = output.z_sample[1:2]    # Second example
        
        # Create interpolation points
        alphas = torch.linspace(0, 1, n_interp).unsqueeze(-1).to(device)
        z_interp = z_start * (1 - alphas) + z_end * alphas
        
        # Decode the interpolation points
        interp_images = model.decode(z_interp)
        
        # Convert to displayable format
        interp_images = (interp_images).clamp(0, 1)
        
        # Also get the original images
        orig_images = (data_batch[:2]).clamp(0, 1)

    # Plot the original and interpolated images
    plt.figure(figsize=(12, 4))
    
    # Plot original start image
    plt.subplot(1, n_interp+2, 1)
    plt.imshow(orig_images[0].permute(1, 2, 0).cpu().numpy())
    plt.title("Original Start")
    plt.axis('off')
    
    # Plot interpolations
    for i in range(n_interp):
        plt.subplot(1, n_interp+2, i+2)
        plt.imshow(interp_images[i].permute(1, 2, 0).cpu().numpy())
        plt.title(f"α={alphas[i].item():.2f}")
        plt.axis('off')
    
    # Plot original end image
    plt.subplot(1, n_interp+2, n_interp+2)
    plt.imshow(orig_images[1].permute(1, 2, 0).cpu().numpy())
    plt.title("Original End")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('convvae_cifar_interpolation.png')
    plt.close()
    
    # Grid of random samples
    n_grid = 3
    with torch.no_grad():
        z = torch.randn(n_grid * n_grid, model.latent_dim).to(device)
        samples = model.decode(z)
        samples = (samples).clamp(0, 1)
    
    # Create grid of images
    grid = torchvision.utils.make_grid(samples, nrow=n_grid, padding=2)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title("Random Samples from Latent Space")
    plt.axis('off')
    plt.savefig('convvae_cifar_random_samples.png')
    plt.close()


def predict_new(hidden_dims,latent_dim,model_path='convvae_cifar_animals_256d.pt', n_grid=3, device=None):
    """
    Load a pretrained ConvVAE model and generate random images from the latent space.
    
    Args:
        model_path (str): Path to the saved model weights file
        n_grid (int): Grid size for visualization (n_grid × n_grid images)
        device: Device to run inference on. If None, will use CUDA if available.
        
    Returns:
        None (saves the generated image grid to disk)
    """
    import torch
    import torchvision
    import matplotlib.pyplot as plt
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define model hyperparameters (must match the trained model)
    in_channels = 3
    img_size = 32
    hidden_dims = hidden_dims
    latent_dim = latent_dim  # This should match the value used in training
    
    # Initialize model with the same architecture as during training
    model = ConvVAE(
        in_channels=in_channels,
        img_size=img_size,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim
    ).to(device)
    
    # Load the saved weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Set model to evaluation mode
    model.eval()
    
    # Generate random samples from the latent space
    with torch.no_grad():
        # Create random latent vectors
        z = torch.randn(n_grid * n_grid, model.latent_dim).to(device)
        
        # Decode the latent vectors to generate images
        samples = model.decode(z)
        
        # Adjust pixel values to [0, 1] range
        # Note: This assumes your model outputs are normalized to [-0.5, 0.5] range
        samples = (samples).clamp(0, 1)
    
    # Create grid of images
    grid = torchvision.utils.make_grid(samples, nrow=n_grid, padding=2)
    
    # Visualize and save the grid
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title("Random Samples from Latent Space")
    plt.axis('off')
    
    # Save the figure
    plt.show()
    output_path = 'convvae_cifar_random_samples3.png'
    plt.savefig(output_path)
    plt.close()
    print(f"Generated images saved to {output_path}")
    
    return grid  # Return the grid tensor in case it's needed for further processing

'''
def latent_space_exploration(
    model_path: str = 'convvae_cifar_animals_64d.pt',
    in_channels: int = 3,
    img_size: int = 32,
    hidden_dims = [32, 64, 128, 256],
    latent_dim: int = 64,
    n_dims: int = 5,    # Number of dimensions to explore
    n_steps: int = 7    # Steps per dimension
):
    """
    Explore how varying individual dimensions in the latent space affects the generated images.
    
    Args:
        model_path (str): Path to the saved model.
        in_channels (int): Number of input channels.
        img_size (int): Size of the input images.
        hidden_dims (list): List of hidden dimensions for the convolutional layers.
        latent_dim (int): Dimensionality of the latent space.
        n_dims (int): Number of dimensions to explore.
        n_steps (int): Number of steps to take in each dimension.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model with the same architecture as during training
    model = ConvVAE(
        in_channels=in_channels,
        img_size=img_size,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim
    ).to(device)
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Select random dimensions to explore (or first n_dims)
    explore_dims = list(range(n_dims))
    
    # Create base latent vector that we'll modify
    z_base = torch.zeros(1, latent_dim, device=device)
    
    # Create figure
    fig, axs = plt.subplots(n_dims, n_steps, figsize=(2*n_steps, 2*n_dims))
    
    # Range for latent traversal
    z_range = torch.linspace(-3.0, 3.0, n_steps)
    
    with torch.no_grad():
        # For each dimension
        for i, dim in enumerate(explore_dims):
            # For each value in the range
            for j, val in enumerate(z_range):
                # Create a copy of the base vector and modify the dimension
                z = z_base.clone()
                z[0, dim] = val
                
                # Generate image
                img = model.decode(z)
                img = (img + 0.5).clamp(0, 1)
                
                # Plot
                axs[i, j].imshow(img[0].permute(1, 2, 0).cpu().numpy())
                axs[i, j].set_title(f"z{dim}={val:.1f}")
                axs[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig('convvae_cifar_latent_traversal.png')
    plt.show()
'''