import torch
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Subset, DataLoader
import torchvision
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

def prepare_data(batch_size=128):
    """
    Prepares a CIFAR-10 dataset filtered to include only 5 animal classes:
    cat (label 3), deer (4), dog (5), frog (6), and horse (7).
    Images are transformed to float tensors with data augmentation for training.
    The data augmentation doesn't actually create new persistent samples
    Your dataset size remains the same (number of original images)
    But effectively, your model sees different variations of each image across epochs
    If you have 10,000 original images and train for 25 epochs, the model potentially sees 250,000 unique variations
    """
    # Define training transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
    ])
    
    # Test transform (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load CIFAR-10 dataset with augmentation for training
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=train_transform,
    )
    
    # Specify the animal classes we want (using the CIFAR-10 class indices)
    # 3: cat, 4: deer, 5: dog, 6: frog, 7: horse
    target_classes = {3, 5}

    # Create a subset of indices belonging only to the target classes
    train_indices = [i for i, (_, label) in enumerate(train_dataset) if label in target_classes]
    animal_train_dataset = Subset(train_dataset, train_indices)
    
    # Create data loader for training with pin_memory for faster GPU transfer
    train_loader = DataLoader(
        animal_train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=True
    )
    
    # For testing, we use the test split with no augmentation
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=test_transform,
    )
    test_indices = [i for i, (_, label) in enumerate(test_dataset) if label in target_classes]
    animal_test_dataset = Subset(test_dataset, test_indices)
    test_loader = DataLoader(
        animal_test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True,
        num_workers=4
    )
    print(f"Train dataset size: {len(animal_train_dataset)}")
    print(f"Test dataset size: {len(animal_test_dataset)}")
    return train_loader, test_loader

def train(model, dataloader, optimizer, prev_updates, device, batch_size, writer=None, scheduler=None, beta_warmup=None):
    """
    Trains the ConvVAE model on the given data with improved training process.
    
    Args:
        model (nn.Module): The ConvVAE model to train.
        dataloader (torch.utils.data.DataLoader): The data loader.
        optimizer: The optimizer.
        prev_updates (int): Number of previous updates.
        device (torch.device): Device to train on.
        batch_size (int): Batch size.
        writer (SummaryWriter, optional): TensorBoard writer.
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler.
        beta_warmup (callable, optional): Function to calculate beta based on update count.
        
    Returns:
        int: Number of updates.
    """
    model.train()  # Set the model to training mode
    running_loss = 0.0
    running_recon_loss = 0.0
    running_kl_loss = 0.0
    
    for batch_idx, (data, _) in enumerate(tqdm(dataloader, desc="Training")):
        n_upd = prev_updates + batch_idx
        
        # Move data to device
        data = data.to(device)
        
        # Apply beta warmup if provided
        if beta_warmup is not None:
            model.beta = beta_warmup(n_upd)
        
        optimizer.zero_grad()  # Zero the gradients
        
        output = model(data)  # Forward pass
        loss = output.loss
        
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        
        optimizer.step()  # Update the model parameters
        
        # Update running losses
        running_loss += loss.item()
        running_recon_loss += output.loss_recon.item()
        running_kl_loss += output.loss_kl.item()
        
        # Log periodically
        if (batch_idx + 1) % 50 == 0:
            # Calculate average losses
            avg_loss = running_loss / 50
            avg_recon_loss = running_recon_loss / 50
            avg_kl_loss = running_kl_loss / 50
            
            # Calculate gradient norms
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            
            # Reset running losses
            running_loss = 0.0
            running_recon_loss = 0.0
            running_kl_loss = 0.0
            
            # Print progress
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Step {n_upd:,} (N samples: {n_upd*batch_size:,}), Loss: {avg_loss:.4f} '
                  f'(Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}) '
                  f'Grad: {total_norm:.4f}, LR: {current_lr:.6f}, Beta: {model.beta:.4f}')

            if writer is not None:
                global_step = n_upd
                writer.add_scalar('Loss/Train', avg_loss, global_step)
                writer.add_scalar('Loss/Train/Reconstruction', avg_recon_loss, global_step)
                writer.add_scalar('Loss/Train/KLD', avg_kl_loss, global_step)
                writer.add_scalar('Training/GradNorm', total_norm, global_step)
                writer.add_scalar('Training/LearningRate', current_lr, global_step)
                writer.add_scalar('Training/Beta', model.beta, global_step)
                
                # Log images periodically
                if (batch_idx + 1) % 200 == 0:
                    recon_images = output.x_recon.clamp(0, 1)
                    orig_images = data.clamp(0, 1)
                    writer.add_images('Train/Reconstructions', recon_images[:8], global_step)
                    writer.add_images('Train/Originals', orig_images[:8], global_step)
    
    # Update learning rate if scheduler is provided
    if scheduler is not None:
        if isinstance(scheduler, ReduceLROnPlateau):
            # This type of scheduler needs the validation loss
            pass  # We'll update it after validation in the main loop
        else:
            scheduler.step()
            
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
        
    Returns:
        float: Average test loss.
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
    print(f'====> Test set loss: {test_loss:.4f} (Recon: {test_recon_loss:.4f}, KLD: {test_kl_loss:.4f})')
    
    if writer is not None:
        writer.add_scalar('Loss/Test', test_loss, global_step=cur_step)
        writer.add_scalar('Loss/Test/Reconstruction', test_recon_loss, global_step=cur_step)
        writer.add_scalar('Loss/Test/KLD', test_kl_loss, global_step=cur_step)
        
        # Log reconstructions and samples
        with torch.no_grad():
            # Get a batch of test data for visualization
            test_batch, _ = next(iter(dataloader))
            test_batch = test_batch.to(device)
            
            # Generate reconstructions
            recon_output = model(test_batch)
            recon_images = recon_output.x_recon.clamp(0, 1)
            orig_images = test_batch.clamp(0, 1)
            
            writer.add_images('Test/Reconstructions', recon_images[:16], global_step=cur_step)
            writer.add_images('Test/Originals', orig_images[:16], global_step=cur_step)
            
            # Log random samples from the latent space
            samples = model.sample(16, device)
            sample_images = samples.clamp(0, 1)
            writer.add_images('Test/Samples', sample_images, global_step=cur_step)
            
            # Log latent space interpolations (between random samples)
            if test_batch.size(0) >= 2:
                # Encode two random images
                mu1, logvar1 = model.encode(test_batch[0:1])
                mu2, logvar2 = model.encode(test_batch[1:2])
                
                # Create interpolation steps
                steps = 8
                z1 = model.reparameterize(mu1, logvar1)
                z2 = model.reparameterize(mu2, logvar2)
                
                # Interpolate in latent space
                z_interp = torch.zeros(steps, model.latent_dim, device=device)
                for i in range(steps):
                    alpha = i / (steps - 1)
                    z_interp[i] = (1 - alpha) * z1 + alpha * z2
                
                # Decode interpolated points
                interp_images = model.decode(z_interp)
                writer.add_images('Test/Interpolations', interp_images.clamp(0, 1), global_step=cur_step)
    
    return test_loss


def create_beta_warmup_fn(warmup_steps=1000, max_beta=1.0):
    """
    Creates a beta warmup function for annealing the KL term weight.
    
    Args:
        warmup_steps (int): Number of steps to reach max_beta.
        max_beta (float): Maximum value of beta.
        
    Returns:
        callable: Function to calculate beta based on update count.
    """
    def beta_warmup_fn(step):
        return min(max_beta, step / warmup_steps * max_beta)
    
    return beta_warmup_fn