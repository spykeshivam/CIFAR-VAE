import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from models.vae import ConvVAE
from train.train_vae import train, test, prepare_data,create_beta_warmup_fn
from utils.model_analysis import visualize_latent_space, interpolate_latent_space, predict_new

def main(latent_dim, hidden_dims):
    # Configuration for ConvVAE with CIFAR-10 images (3x32x32)
    batch_size = 128
    learning_rate = 1e-3
    weight_decay = 1e-2
    num_epochs = 25
    latent_dim = latent_dim  # Adjust the latent space size as needed

    # Beta warmup configuration
    warmup_steps = 2000  # Number of steps for beta to reach max value
    max_beta = 1.0       # Maximum value for beta
    beta_warmup_fn = create_beta_warmup_fn(warmup_steps, max_beta)
    
    # ConvVAE architecture configuration
    in_channels = 3  # RGB images
    img_size = 32    # CIFAR-10 images are 32x32
    hidden_dims = hidden_dims  # Encoder/decoder channel dimensions
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_loader, test_loader = prepare_data(batch_size)
    
    # Initialize ConvVAE with appropriate parameters
    model = ConvVAE(
        in_channels=in_channels,
        img_size=img_size,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        beta=0.0  # Start with beta=0, will be controlled by warmup function
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    writer = SummaryWriter(f'runs/cifar_animals/convvae_{datetime.now().strftime("%Y%m%d-%H%M%S")}')
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params:,}")
    print(model)
    
    prev_updates = 0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        #prev_updates = train(model, train_loader, optimizer, prev_updates, device, batch_size, writer=writer)
        # Pass the beta_warmup_fn to the train function
        prev_updates = train(
            model,
            train_loader,
            optimizer,
            prev_updates,
            device,
            batch_size,
            writer=writer,
            scheduler=scheduler,
            beta_warmup=beta_warmup_fn
        )
        
        # Test the model and get the test loss
        #test(model, test_loader, prev_updates, device, writer=writer)  # Updated test function signature
        test_loss = test(model, test_loader, prev_updates, device, writer=writer)
        
        # Update the scheduler if using ReduceLROnPlateau
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(test_loss)
    
    # Save the model at the end of training
    torch.save(model.state_dict(), f'convvae_cifar_animals_{latent_dim}d.pt')
    print("Training completed and model saved!")
    
    # Optional model analysis - may need to be updated for convolutional architecture
    try:
        # Visualize latent space (update these functions for ConvVAE)
        visualize_latent_space(model, test_loader, device)
        interpolate_latent_space(model, test_loader, device)
    except Exception as e:
        print(f"Could not run visualization functions: {e}")

if __name__ == "__main__":
    hidden_dims = [32, 64,64, 128,128, 256]
    latent_dim = 128
    #main(latent_dim, hidden_dims)
    predict_new(latent_dim=latent_dim, hidden_dims=hidden_dims, model_path='convvae_cifar_animals_128d.pt', n_grid=3)