import torch
from tqdm import tqdm
from torchvision.transforms import v2
from torchvision import transforms
from torch.utils.data import Subset
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.vae import ConvVAE

def prepare_data(batch_size=128):
    """
    Prepares a CIFAR-10 dataset filtered to include only 5 animal classes:
    cat (label 3), deer (4), dog (5), frog (6), and horse (7).
    Images are transformed to float tensors in range [-0.5, 0.5].
    """
    # Define the transforms
    # In prepare_data function
    transform = transforms.Compose([
        transforms.ToTensor(),  # Already gives [0,1] range
    ])

    # Load CIFAR-10 dataset
    dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform,
    )

    # Specify the animal classes we want (using the CIFAR-10 class indices)
    # 3: cat, 4: deer, 5: dog, 6: frog, 7: horse
    target_classes = {3, 4, 5, 6, 7}

    # Create a subset of indices belonging only to the target classes
    indices = [i for i, (_, label) in enumerate(dataset) if label in target_classes]
    animal_dataset = Subset(dataset, indices)
    
    # Create data loader for training
    train_loader = DataLoader(animal_dataset, batch_size=batch_size, shuffle=True)
    
    # For testing, we can do the same with the test split
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform,
    )
    test_indices = [i for i, (_, label) in enumerate(test_dataset) if label in target_classes]
    animal_test_dataset = Subset(test_dataset, test_indices)
    test_loader = DataLoader(animal_test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train(model, dataloader, optimizer, prev_updates, device, batch_size, writer=None):
    """
    Trains the ConvVAE model on the given data.
    
    Args:
        model (nn.Module): The ConvVAE model to train.
        dataloader (torch.utils.data.DataLoader): The data loader.
        optimizer: The optimizer.
        prev_updates (int): Number of previous updates.
        device (torch.device): Device to train on.
        batch_size (int): Batch size.
        writer (SummaryWriter, optional): TensorBoard writer.
        
    Returns:
        int: Number of updates.
    """
    model.train()  # Set the model to training mode
    
    for batch_idx, (data, _) in enumerate(tqdm(dataloader, desc="Training")):
        n_upd = prev_updates + batch_idx
        
        # Move data to device - no need to flatten for ConvVAE
        data = data.to(device)
        
        optimizer.zero_grad()  # Zero the gradients
        
        output = model(data)  # Forward pass
        loss = output.loss
        
        loss.backward()
        
        if n_upd % 100 == 0:
            # Calculate and log gradient norms
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
        
            print(f'Step {n_upd:,} (N samples: {n_upd*batch_size:,}), Loss: {loss.item():.4f} '
                  f'(Recon: {output.loss_recon.item():.4f}, KL: {output.loss_kl.item():.4f}) '
                  f'Grad: {total_norm:.4f}')

            if writer is not None:
                global_step = n_upd
                writer.add_scalar('Loss/Train', loss.item(), global_step)
                writer.add_scalar('Loss/Train/BCE', output.loss_recon.item(), global_step)
                writer.add_scalar('Loss/Train/KLD', output.loss_kl.item(), global_step)
                writer.add_scalar('GradNorm/Train', total_norm, global_step)
                
                # Log a few reconstructed images periodically
                if n_upd % 500 == 0:
                    recon_images = (output.x_recon + 0.5).clamp(0, 1)  # Convert back to [0,1] range
                    orig_images = (data + 0.5).clamp(0, 1)
                    writer.add_images('Train/Reconstructions', recon_images[:8], global_step)
                    writer.add_images('Train/Originals', orig_images[:8], global_step)
            
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)    
        
        optimizer.step()  # Update the model parameters
        
    return prev_updates + len(dataloader)


def test(model, dataloader, cur_step, device, writer=None):
    """
    Tests the ConvVAE model on the given data.
    
    Args:
        model (nn.Module): The ConvVAE model to test.
        dataloader (torch.utils.data.DataLoader): The data loader.
        cur_step (int): The current step.
        device (torch.device): Device to test on.
        writer (SummaryWriter, optional): TensorBoard writer.
    """
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    test_recon_loss = 0
    test_kl_loss = 0
    
    with torch.no_grad():
        for data, _ in tqdm(dataloader, desc='Testing'):
            data = data.to(device)
            
            output = model(data, compute_loss=True)  # Forward pass
            
            test_loss += output.loss.item()
            test_recon_loss += output.loss_recon.item()
            test_kl_loss += output.loss_kl.item()
            
    test_loss /= len(dataloader)
    test_recon_loss /= len(dataloader)
    test_kl_loss /= len(dataloader)
    print(f'====> Test set loss: {test_loss:.4f} (BCE: {test_recon_loss:.4f}, KLD: {test_kl_loss:.4f})')
    
    if writer is not None:
        writer.add_scalar('Loss/Test', test_loss, global_step=cur_step)
        writer.add_scalar('Loss/Test/BCE', test_recon_loss, global_step=cur_step)
        writer.add_scalar('Loss/Test/KLD', test_kl_loss, global_step=cur_step)
        
        # Log reconstructions
        with torch.no_grad():
            # Get a batch of test data for visualization
            test_batch, _ = next(iter(dataloader))
            test_batch = test_batch.to(device)
            
            # Generate reconstructions
            recon_output = model(test_batch)
            recon_images = (recon_output.x_recon + 0.5).clamp(0, 1)
            orig_images = (test_batch + 0.5).clamp(0, 1)
            
            writer.add_images('Test/Reconstructions', recon_images[:16], global_step=cur_step)
            writer.add_images('Test/Originals', orig_images[:16], global_step=cur_step)
            
            # Log random samples from the latent space
            samples = model.sample(16, device)
            sample_images = (samples + 0.5).clamp(0, 1)
            writer.add_images('Test/Samples', sample_images, global_step=cur_step)

# Main training function
def train_vae(epochs=50, batch_size=128, learning_rate=1e-3):
    """
    Main function to train the ConvVAE model.
    
    Args:
        epochs (int): Number of epochs to train for.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for optimizer.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare data
    train_loader, test_loader = prepare_data(batch_size=batch_size)
    
    # CIFAR-10 images are 3x32x32
    model = ConvVAE(
        in_channels=3,
        img_size=32,
        hidden_dims=[32, 64, 128, 256],
        latent_dim=128
    ).to(device)
    
    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Set up TensorBoard writer
    writer = SummaryWriter(log_dir='runs/convvae_cifar10_animals')
    
    # Initialize updates counter
    updates = 0
    
    # Training loop
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        
        # Train
        updates = train(model, train_loader, optimizer, updates, device, batch_size, writer)
        
        # Test
        test(model, test_loader, updates, device, writer)
        
        # Save model checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'updates': updates,
            'epoch': epoch,
        }, f'checkpoints/convvae_epoch_{epoch}.pt')
    
    writer.close()
    print("Training complete!")
    
    return model