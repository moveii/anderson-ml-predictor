# Patrick Berger, 2024

import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np
from typing import Dict, Union

from model import AutoregressiveTransformer, ModelConfig
from dataset import ImpurityDataset
from columns import encoder_input, labels

# Configuration
CONFIG: Dict[str, Union[int, float, str, bool]] = {
    "seed": 42,
    "dataset_path": "data/data_50k.csv",
    "model_checkpoint": "model/model-2024-09-19.pth",
    "batch_size": 32,
    "validation_size": 0.1,
    "test_size": 0.1,
    "use_scaling": True,
    "pair_up_labels": False,
}

# Model configuration (ensure this matches your trained model)
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


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(config, device):
    model = AutoregressiveTransformer(MODEL_CONFIG, device).to(device)
    model.load_state_dict(torch.load(config["model_checkpoint"], map_location=device))
    return model


def create_test_dataloader(config):
    df = pd.read_csv(config["dataset_path"])

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

    return DataLoader(dataset.get_test_dataset(), batch_size=config["batch_size"], shuffle=False), dataset


def sample(model, encoder_input, targets, num_initial_targets, max_length, pair_up_labels, device):
    model.eval()
    encoder_input = encoder_input.to(device)

    with torch.no_grad():
        encoder_output = model.encode(encoder_input)

        start_token = [0, 0] if pair_up_labels else [0]
        start_token_tensor = torch.tensor([start_token], dtype=torch.float).to(device)
        decoder_input = start_token_tensor.expand(encoder_input.size(0), 1, -1)

        if num_initial_targets > 0:
            decoder_input = torch.cat((decoder_input, targets[:, :num_initial_targets, :]), dim=1)

        while decoder_input.size(1) < max_length + 1:  # +1 for the start token
            output = model.decode(decoder_input, encoder_output)
            next_token = output[:, -1:, :]
            decoder_input = torch.cat((decoder_input, next_token), dim=1)

    return decoder_input[:, 1:, :]


def evaluate_autoregressive(model, test_loader, criterion, pair_up_labels, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for encoder_input, _, targets in test_loader:

            outputs = sample(model, encoder_input, targets, 0, targets.size(1), pair_up_labels, device)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    return total_loss / len(test_loader)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():

        for encoder_input, decoder_input, targets in loader:

            outputs = model(encoder_input, decoder_input)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    return total_loss / len(test_loader)


set_seed(CONFIG["seed"])
device = get_device()
print(f"Using device: {device}")

model = load_model(CONFIG, device)

test_loader, dataset = create_test_dataloader(CONFIG)
criterion = MAPELoss(dataset.label_scaler).to(device)

test_losss = evaluate(model, test_loader, criterion)
print(f"Test Loss: {test_losss:.4f}")

test_ar_losss = evaluate_autoregressive(model, test_loader, criterion, CONFIG["pair_up_labels"], device)
print(f"Autoregressive Test Loss: {test_ar_losss:.4f}")
