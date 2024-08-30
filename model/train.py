import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import train_test_split

from model import AutoregressiveTransformer, ModelConfig
from dataset import ImpurityDataset
from columns import encoder_input, labels

# Configuration dictionary
CONFIG: Dict[str, Union[int, float, str, bool]] = {
    "seed": 42,
    "file_path": "data/data_10k.csv",
    "batch_size": 32,
    "validation_size": 0.1,
    "use_scaling": True,
    "pair_up_columns": True,
    "num_epochs": 150,
    "learning_rate": 1e-4,
    "scheduler_gamma": 0.98,
}

# Model configuration
MODEL_CONFIG = ModelConfig(
    output_dim=2 if CONFIG["pair_up_columns"] else 1,
    d_model=256,
    encoder_max_seq_length=len(encoder_input),
    encoder_input_dim=1,
    encoder_dim_feedforward=256 * 4,
    encoder_nhead=4,
    encoder_num_layers=4,
    decoder_max_seq_length=len(labels) // (2 if CONFIG["pair_up_columns"] else 1),
    decoder_input_dim=2 if CONFIG["pair_up_columns"] else 1,
    decoder_dim_feedforward=256 * 4,
    decoder_nhead=4,
    decoder_num_layers=4,
    dropout=0.1,
    activation="gelu",
    bias=True,
)


class MAPELoss(nn.Module):
    """Custom Mean Absolute Percentage Error (MAPE) loss function."""

    def __init__(self, scaler: StandardScaler, epsilon: float = 1e-8):
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
        outputs_reshaped = outputs.view(B, T * C)
        targets_reshaped = targets.view(B, T * C)

        scale = torch.tensor(self.scaler.scale_).to(outputs.device)
        mean = torch.tensor(self.scaler.mean_).to(outputs.device)

        outputs_unscaled = outputs_reshaped * scale + mean
        targets_unscaled = targets_reshaped * scale + mean

        outputs_unscaled = outputs_unscaled.view(B, T, C)
        targets_unscaled = targets_unscaled.view(B, T, C)

        diff = torch.abs(targets_unscaled - outputs_unscaled)
        norm = torch.norm(targets_unscaled, p=2, dim=1, keepdim=True)
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


def load_data(file_path: str, encoder_input: List[str], labels: List[str]) -> pd.DataFrame:
    """Load and preprocess the data from a CSV file."""
    df = pd.read_csv(file_path)
    return df[encoder_input + labels]


def create_datasets(df: pd.DataFrame, config: Dict[str, Union[int, float, str, bool]]) -> Tuple[ImpurityDataset, Subset, Subset]:
    """Create train and validation datasets."""
    dataset = ImpurityDataset(
        df,
        encoder_input,
        labels,
        config["pair_up_columns"],
        config["use_scaling"],
        config["validation_size"],
        device=get_device(),
        seed=config["seed"],
    )

    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=config["validation_size"], random_state=config["seed"])

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    return dataset, train_dataset, val_dataset


def create_dataloaders(train_dataset: Subset, val_dataset: Subset, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Create DataLoader objects for training and validation sets."""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def create_model(config: ModelConfig, device: torch.device) -> AutoregressiveTransformer:
    """Create and initialize the model."""
    model = AutoregressiveTransformer(config, device).to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    return model


def create_optimizer_and_scheduler(
    model: nn.Module, config: Dict[str, Union[int, float, str, bool]]
) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
    """Create optimizer and learning rate scheduler."""
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["scheduler_gamma"])
    return optimizer, scheduler


def train_epoch(
    model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, device: torch.device
) -> float:
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


def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    """Validate the model."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for encoder_input, decoder_input, targets in loader:
            outputs = model(encoder_input, decoder_input)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    return total_loss / len(loader)


def reverse_transform_output(data: torch.Tensor, scaler: StandardScaler) -> torch.Tensor:
    """Reverse the scaling transformation on the output data."""
    B, T, C = data.shape
    data = data.view(B, T * C)
    data = data.cpu().numpy()
    return torch.tensor(scaler.inverse_transform(data)).view(B, T, C)


def validate_mape(
    model: nn.Module, loader: DataLoader, label_scaler: StandardScaler, use_scaling: bool = True, epsilon: float = 1e-8
) -> float:
    """Validate the model using Mean Absolute Percentage Error (MAPE)."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for encoder_input, decoder_input, targets in loader:
            outputs = model(encoder_input, decoder_input)

            if use_scaling:
                outputs = reverse_transform_output(outputs, label_scaler)
                targets = reverse_transform_output(targets, label_scaler)

            ape = torch.abs(targets - outputs) / (torch.norm(targets, p=2, dim=1, keepdim=True) + epsilon)
            mape = torch.mean(ape) * 100
            total_loss += mape.item()

    return total_loss / len(loader)


def sample(
    model: nn.Module, encoder_input: torch.Tensor, max_length: int, device: torch.device, start_token: List[float] = [0, 0]
) -> torch.Tensor:
    """Generate a sample output sequence using the trained model."""
    model.eval()
    encoder_input = encoder_input.to(device)

    with torch.no_grad():
        encoder_output = model.encode(encoder_input)

        start_token_tensor = torch.tensor([start_token], dtype=torch.float).to(device)
        decoder_input = start_token_tensor.expand(encoder_input.size(0), 1, -1)

        for _ in range(max_length):
            output = model.decode(decoder_input, encoder_output)
            next_token = output[:, -1:, :]
            decoder_input = torch.cat((decoder_input, next_token), dim=1)

    return decoder_input[:, 1:, :]


def validate_autoregressive_mape(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    label_scaler: StandardScaler,
    use_scaling: bool = True,
    epsilon: float = 1e-8,
) -> Tuple[float, List[float]]:
    """Validate the model using autoregressive prediction and MAPE."""
    model.eval()
    losses = []

    with torch.no_grad():
        for encoder_input, _, targets in val_loader:
            outputs = sample(model, encoder_input, len(labels) // (2 if CONFIG["pair_up_columns"] else 1), device)

            if use_scaling:
                outputs = reverse_transform_output(outputs, label_scaler)
                targets = reverse_transform_output(targets, label_scaler)

            ape = torch.abs(targets - outputs) / (torch.norm(targets, p=2, dim=1, keepdim=True) + epsilon)
            mape = torch.mean(ape) * 100
            losses.append(mape.item())

    return np.average(losses), losses


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    criterion: nn.Module,
    config: Dict[str, Union[int, float, str, bool]],
    label_scaler: StandardScaler,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Train the model and return training history."""
    train_losses, val_losses = [], []
    train_mapes, val_mapes = [], []
    best_val_loss = float("inf")

    for epoch in range(config["num_epochs"]):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, get_device())
        val_loss = validate(model, val_loader, criterion, get_device())

        train_mape = validate_mape(model, train_loader, label_scaler)
        val_mape = validate_mape(model, val_loader, label_scaler)

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_mapes.append(train_mape)
        val_mapes.append(val_mape)

        print(
            f"Epoch {epoch+1}/{config['num_epochs']}, "
            f"Train Loss: {train_loss:.4f}, Train MAPE: {train_mape:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val MAPE: {val_mape:.2f}%, "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")

    return train_losses, val_losses, train_mapes, val_mapes


def plot_losses(train_losses: List[float], val_losses: List[float], train_mapes: List[float], val_mapes: List[float]) -> None:
    """Plot and save training and validation loss curves."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

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

    ax3.plot(train_mapes, label="Train MAPE")
    ax3.plot(val_mapes, label="Validation MAPE")
    ax3.set_title("MAPE Over Epochs")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("MAPE (%)")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig("training_plots.png")
    plt.close()


set_seed(CONFIG["seed"])
device = get_device()

df = load_data(CONFIG["file_path"], encoder_input, labels)
dataset, train_dataset, val_dataset = create_datasets(df, CONFIG)
train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, CONFIG["batch_size"])

model = create_model(MODEL_CONFIG, device)
optimizer, scheduler = create_optimizer_and_scheduler(model, CONFIG)
criterion = MAPELoss(dataset.label_scaler).to(device)

train_losses, val_losses, train_mapes, val_mapes = train_model(
    model, train_loader, val_loader, optimizer, scheduler, criterion, CONFIG, dataset.label_scaler
)

plot_losses(train_losses, val_losses, train_mapes, val_mapes)

auto_mape = validate_autoregressive_mape(model, val_loader, device, dataset.label_scaler)
print(f"Autoregressive Validation MAPE: {auto_mape:.2f}%")
