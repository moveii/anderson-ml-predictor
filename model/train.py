# Patrick Berger, 2024

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from model import AutoregressiveTransformer, ModelConfig
from dataset import ImpurityDataset
from columns import encoder_input, labels

# Overall configuration
CONFIG: Dict[str, Union[int, float, str, bool]] = {
    "seed": 42,
    "dataset_path": "data/data_50k.csv",
    "model_checkpoint": "model/model-2024-09-24-hybridization-approximations.pth",
    "pretrained_model": None,  # Set this to the path of a pretrained model to load, or None for a fresh start
    "batch_size": 32,
    "validation_size": 0.1,
    "test_size": 0.1,
    "use_scaling": True,
    "pair_up_labels": False,  # whether to pair up the labels, so t(i) and t(max-i) are predicted at the same time
    "num_epochs": 500,
    "learning_rate": 1e-4,
    "scheduler_gamma": 0.99,
}

# Model configuration
MODEL_CONFIG = ModelConfig(
    output_dim=2 if CONFIG["pair_up_labels"] else 1,
    d_model=256,
    encoder_max_seq_length=len(encoder_input),
    encoder_input_dim=1,
    encoder_dim_feedforward=256 * 4,
    encoder_nhead=4,
    encoder_num_layers=4,
    decoder_max_seq_length=len(labels) // (2 if CONFIG["pair_up_labels"] else 1),
    decoder_input_dim=2 if CONFIG["pair_up_labels"] else 1,
    decoder_dim_feedforward=256 * 4,
    decoder_nhead=4,
    decoder_num_layers=4,
    dropout=0.1,
    activation="gelu",
    bias=True,
)


class MAPELoss(nn.Module):
    """Custom Mean Absolute Percentage Error (MAPE) loss function."""

    def __init__(self, scaler, epsilon: float = 1e-8):
        super(MAPELoss, self).__init__()
        self.scaler = scaler
        self.epsilon = epsilon

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the MAPE loss.

        Args:
            outputs (torch.Tensor): Predicted values
            targets (torch.Tensor): True values

        Returns:
            torch.Tensor: Computed MAPE loss
        """
        B, T, C = outputs.shape
        device = outputs.device

        outputs = outputs.view(B, T * C)
        targets = targets.view(B, T * C)

        if self.scaler != None:
            scale = torch.tensor(self.scaler.scale_).to(device)
            mean = torch.tensor(self.scaler.mean_).to(device)

            outputs = outputs * scale + mean
            targets = targets * scale + mean

            outputs = outputs.view(B, T, C)
            targets = targets.view(B, T, C)

        diff = torch.abs(targets - outputs)
        norm = torch.norm(targets, p=2, dim=1, keepdim=True)
        ape = diff / (norm + self.epsilon)

        return torch.mean(ape) * 100


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get the available device (CPU or GPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def load_data(dataset_path: str, encoder_input: List[str], labels: List[str]) -> pd.DataFrame:
    """Load and preprocess the data from a CSV file."""
    df = pd.read_csv(dataset_path)
    return df[encoder_input + labels]


def create_datasets(df: pd.DataFrame, config: Dict[str, Union[int, float, str, bool]]) -> Tuple[ImpurityDataset, Subset, Subset]:
    """Create train and validation datasets."""
    dataset = ImpurityDataset(
        df[encoder_input + labels],
        encoder_input,
        labels,
        config["pair_up_labels"],
        config["use_scaling"],
        config["validation_size"],
        config["test_size"],
        device=get_device(),
        seed=config["seed"],
    )

    return dataset, dataset.get_train_dataset(), dataset.get_val_dataset()


def create_dataloaders(train_dataset: Subset, val_dataset: Subset, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Create DataLoader objects for training and validation sets."""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return train_loader, val_loader


def create_model(config: ModelConfig, device: torch.device, pretrained_path: str = None) -> AutoregressiveTransformer:
    """Create and initialize the model, optionally loading from a pretrained file."""
    model = AutoregressiveTransformer(config, device).to(device)

    if pretrained_path and os.path.isfile(pretrained_path):
        print(f"Loading pretrained model from {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
    else:
        print("Initializing fresh model")

    print(f"Model has {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    return model


def create_optimizer_and_scheduler(
    model: nn.Module, config: Dict[str, Union[int, float, str, bool]]
) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
    """Create optimizer and learning rate scheduler."""
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["scheduler_gamma"])
    return optimizer, scheduler


def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module) -> float:
    """Train the model for one epoch."""
    model.train()
    total_loss = 0

    for encoder_input, decoder_input, targets in train_loader:
        optimizer.zero_grad()

        outputs = model(encoder_input, decoder_input)
        loss = criterion(outputs, targets)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> float:
    """Validate the model."""
    model.eval()
    total_loss = 0

    with torch.no_grad():

        for encoder_input, decoder_input, targets in loader:

            outputs = model(encoder_input, decoder_input)
            loss = criterion(outputs, targets)

            total_loss += loss.item()

    return total_loss / len(loader)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    criterion: nn.Module,
    config: Dict[str, Union[int, float, str, bool]],
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Train the model and return training history."""
    train_losses, val_losses = [], []

    lowest_train_loss = float("inf")
    lowest_val_loss = float("inf")

    best_train_epoch = -1
    best_val_epoch = -1

    for epoch in range(config["num_epochs"]):
        train_epoch(model, train_loader, optimizer, criterion)

        train_loss = validate(model, train_loader, criterion)
        val_loss = validate(model, val_loader, criterion)

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch+1}/{config['num_epochs']}, "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        if train_loss < lowest_train_loss:
            lowest_train_loss = train_loss
            best_train_epoch = epoch + 1

        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            best_val_epoch = epoch + 1
            torch.save(model.state_dict(), config["model_checkpoint"])

    print(f"Lowest Train Loss of {lowest_train_loss:.6f} at Epoch {best_train_epoch}")
    print(f"Lowest Val Loss of {lowest_val_loss:.6f} at Epoch {best_val_epoch}")

    return train_losses, val_losses


def plot_losses(train_losses: List[float], val_losses: List[float]) -> None:
    """Plot and save training and validation loss curves."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))

    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses, label="Validation Loss")
    ax1.set_yscale("log")
    ax1.set_title("Loss Over Epochs (Log Scale)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(train_losses, label="Train Loss")
    ax2.plot(val_losses, label="Validation Loss")
    ax2.set_title("Loss Over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("plots/training_plots.png")
    plt.close()


set_seed(CONFIG["seed"])
device = get_device()

df = load_data(CONFIG["dataset_path"], encoder_input, labels)
dataset, train_dataset, val_dataset = create_datasets(df, CONFIG)
train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, CONFIG["batch_size"])

model = create_model(MODEL_CONFIG, device, CONFIG["pretrained_model"])
optimizer, scheduler = create_optimizer_and_scheduler(model, CONFIG)
criterion = MAPELoss(dataset.label_scaler).to(device)

train_losses, val_losses = train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, CONFIG)

plot_losses(train_losses, val_losses)
