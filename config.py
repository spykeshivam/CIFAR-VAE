from dataclasses import dataclass

@dataclass
class Config:
    input_dim: int = 784
    hidden_dim: int = 400
    latent_dim: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    epochs: int = 10
    weight_decay = 1e-2
    num_epochs: int = 10