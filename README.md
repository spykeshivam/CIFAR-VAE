# ConvVAE CIFAR-10 Animals

A Convolutional Variational Autoencoder (ConvVAE) implementation for image generation and manipulation using the animal classes from the CIFAR-10 dataset.

## Project Overview

This project implements a Convolutional Variational Autoencoder (VAE) to learn latent representations of animal images from the CIFAR-10 dataset. The model focuses on 5 animal classes: cats, deer, dogs, frogs, and horses. The VAE can generate new animal images, interpolate between existing images, and be used for various other image manipulation tasks.

## Features

- Deep convolutional architecture for image encoding and decoding
- Customizable latent space dimensionality
- Training with KL divergence and reconstruction loss
- TensorBoard integration for visualization of training progress
- Tools for latent space exploration and image generation
- Model checkpointing for easy resuming of training

## Installation

```bash
# Clone the repository
git clone https://github.com/spykeshivam/CIFAR-VAE.git
cd CIFAR-VAE

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
project/
│
├── main.py                    # Main entry point for training and evaluation
├── models/
│   └── vae.py                 # ConvVAE model definition
├── train/
│   └── train_vae.py           # Training and testing functions
├── utils/
│   └── model_analysis.py      # Functions for latent space visualization
├── runs/                      # TensorBoard logs (created during training)
├── checkpoints/               # Saved model weights (created during training)
└── README.md                  # This file
```

## Usage

### Training the Model

To train the model with default parameters:

```bash
python main.py
```
If you just want inference from an existing .pt file, run the predict_new() function
### Customizing Training

Edit the parameters in `main.py` to customize the training process:

```python
# Configuration
batch_size = 128
learning_rate = 1e-3
weight_decay = 1e-2
num_epochs = 50
latent_dim = 64  # Size of the latent space

# ConvVAE architecture
in_channels = 3  # RGB images
img_size = 32    # CIFAR-10 images are 32x32
hidden_dims = [32, 64, 128, 256]  # Encoder/decoder channel dimensions
```

### Visualizing Training Progress

```bash
tensorboard --logdir=runs
```

Then open http://localhost:6006 in your browser.

## Technical Details

### ConvVAE Architecture

The Convolutional VAE consists of:

1. **Encoder**: Series of convolutional layers that reduce spatial dimensions while increasing feature channels
2. **Latent Space**: Projections to mean and variance, followed by the reparameterization trick
3. **Decoder**: Series of transposed convolutions that increase spatial dimensions while decreasing feature channels

### Data Preprocessing

The project uses the CIFAR-10 dataset and:
- Filters to include only the 5 animal classes (cats, deer, dogs, frogs, horses)
- Transforms images to the range [-0.5, 0.5]
- Applies data augmentation during training

### Loss Function

The VAE is trained with a combined loss:
- Reconstruction loss: Binary cross-entropy (or MSE for grayscale)
- KL divergence: Regularization term to make the latent distribution approach a standard normal distribution

## Troubleshooting

### Common Issues

- **Out of memory errors**: Reduce batch size or model size (hidden dimensions/latent dimension).
- **Slow convergence**: Adjust learning rate or try a different optimizer.

## License

[MIT License](LICENSE)

## Acknowledgments

- The CIFAR-10 dataset is provided by the Canadian Institute For Advanced Research
- PyTorch and TorchVision for the deep learning framework
- TensorBoard for visualization tools