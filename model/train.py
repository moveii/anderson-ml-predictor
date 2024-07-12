import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import train_test_split

from model import AutoregressiveTransformer, ModelConfig
from dataset import ImpurityDataset, compute_scalers

# set the seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

file_path = "data/20230825_144318_10k_EVDoubExp-TExp-wmax5-sparse-hyb_with_perturbation.csv"

# note that 'U' is missing in the encoder input, because its 0 in the original data set, but should be added later
# ReFso should be replaced with the hybridisation function
encoder_input = [
    "beta",
    "ReFso1",
    "ReFso3",
    "ReFso5",
    "ReFso7",
    "ReFso9",
    "ReFso11",
    "ReFso13",
    "ReFso15",
    "ReFso17",
    "ReFso19",
    "ReFso21",
    "ReFso23",
    "ReFso25",
    "ReFso29",
    "ReFso33",
    "ReFso37",
    "ReFso43",
    "ReFso49",
    "ReFso57",
    "ReFso69",
    "ReFso83",
    "ReFso101",
    "ReFso127",
    "ReFso165",
    "ReFso237",
    "ReFso399",
    "ReFso1207",
]

labels = [
    "ReSf1",
    "ImSf1",
    "ReSf3",
    "ImSf3",
    "ReSf5",
    "ImSf5",
    "ReSf7",
    "ImSf7",
    "ReSf9",
    "ImSf9",
    "ReSf11",
    "ImSf11",
    "ReSf13",
    "ImSf13",
    "ReSf15",
    "ImSf15",
    "ReSf17",
    "ImSf17",
    "ReSf19",
    "ImSf19",
    "ReSf21",
    "ImSf21",
    "ReSf23",
    "ImSf23",
    "ReSf25",
    "ImSf25",
    "ReSf29",
    "ImSf29",
    "ReSf33",
    "ImSf33",
    "ReSf37",
    "ImSf37",
    "ReSf43",
    "ImSf43",
    "ReSf49",
    "ImSf49",
    "ReSf57",
    "ImSf57",
    "ReSf69",
    "ImSf69",
    "ReSf83",
    "ImSf83",
    "ReSf101",
    "ImSf101",
    "ReSf127",
    "ImSf127",
    "ReSf165",
    "ImSf165",
    "ReSf237",
    "ImSf237",
    "ReSf399",
    "ImSf399",
    "ReSf1207",
    "ImSf1207",
]


df = pd.read_csv(file_path, skiprows=4)  # we skip the first four lines, because they are just metadata
df = df[encoder_input + labels]

# remove one special row, looks very weird; ReSf1 = 2.377167465976437e-06
df = df[abs(df["ReSf1"]) >= 1e-05]
# sorted(df['ReSf1'].abs())[:10]

validation_size = 0.1  # 90% training, 10% for validation
use_scaling = True

if use_scaling:
    # make sure we use the same seed, otherwise the two splits differ!
    feature_scaler, label_scaler = compute_scalers(df, encoder_input, labels, validation_size, seed)
    dataset = ImpurityDataset(df, encoder_input, labels, feature_scaler, label_scaler, device=device)
else:
    dataset = ImpurityDataset(df, encoder_input, labels, device=device)

indices = list(range(len(dataset)))
# make sure we use the same seed, otherwise the two splits differ!
train_indices, val_indices = train_test_split(indices, test_size=validation_size, random_state=seed)
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# print(dataset.__getitem__(0))

# Model configuration and initialization
config = ModelConfig(
    output_dim=2,
    d_model=512,
    max_sequence_length=len(encoder_input),
    encoder_input_dim=1,
    encoder_dim_feedforward=512 * 4,
    encoder_nhead=4,
    encoder_num_layers=4,
    decoder_input_dim=2,
    decoder_dim_feedforward=512 * 4,
    decoder_nhead=4,
    decoder_num_layers=4,
    dropout=0.1,
    activation="gelu",
    bias=True,
)

model = AutoregressiveTransformer(config, device).to(device)
print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, threshold=5e-6, min_lr=1e-5, factor=0.8)


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    losses = []

    for encoder_input, decoder_input, targets in train_loader:

        optimizer.zero_grad()

        outputs = model(encoder_input, decoder_input, device)
        loss = criterion(outputs, targets)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())

    return np.average(losses)


def validate(model, val_loader, criterion, device):
    model.eval()
    losses = []

    with torch.no_grad():

        for encoder_input, decoder_input, targets in val_loader:

            outputs = model(encoder_input, decoder_input, device)
            loss = criterion(outputs, targets)

            losses.append(loss.item())

    return np.average(losses)


def reverse_transform_output(data, scaler):
    B, T, C = data.shape
    data = data.view(B, T * C)
    data = data.cpu().numpy()
    return scaler.inverse_transform(data)


def validate_mape(model, loader, use_scaling=use_scaling, epsilon=1e-8):
    model.eval()
    losses = []

    with torch.no_grad():

        for encoder_input, decoder_input, targets in loader:
            outputs = model(encoder_input, decoder_input, device)

            if use_scaling:
                outputs = torch.tensor(reverse_transform_output(outputs, label_scaler))
                targets = torch.tensor(reverse_transform_output(targets, label_scaler))

            ape = torch.abs((targets - outputs) / (targets + epsilon))
            mape = torch.mean(ape) * 100

            losses.append(mape.item())

    return np.average(losses)


def sample(model, encoder_input, max_length, device, start_token=[0, 0]):
    model.eval()
    encoder_input = encoder_input.to(device)

    with torch.no_grad():
        encoder_output = model.encode(encoder_input)

        start_token_tensor = torch.tensor([start_token], dtype=torch.float).to(device)
        decoder_input = start_token_tensor.expand(encoder_input.size(0), 1, -1)

        for _ in range(max_length):
            output = model.decode(decoder_input, encoder_output, device)
            next_token = output[:, -1:, :]
            decoder_input = torch.cat((decoder_input, next_token), dim=1)

    return decoder_input[:, 1:, :]


def validate_autoregressive_mape(model, val_loader, device, use_scaling=use_scaling, epsilon=1e-8):
    model.eval()
    losses = []

    with torch.no_grad():

        for encoder_input, _, targets in val_loader:

            outputs = sample(model, encoder_input, 27, device)

            if use_scaling:
                outputs = torch.tensor(reverse_transform_output(outputs, label_scaler))
                targets = torch.tensor(reverse_transform_output(targets, label_scaler))

            ape = torch.abs((targets - outputs) / (targets + epsilon))
            mape = torch.mean(ape) * 100

            losses.append(mape.item())

    return np.average(losses)


print(f"Inititial validation loss: {validate(model, val_loader, criterion, device)}")
print(f"Inititial MAPE validation loss: {validate_mape(model, val_loader)}")
print(f"Inititial Auto regresssive validation loss: {validate_autoregressive_mape(model, val_loader, device)}")

num_epochs = 100

lowest_train_loss = float("inf")
lowest_val_loss = float("inf")
lowest_val_reg_loss = float("inf")

best_train_epoch = -1
best_val_epoch = -1
best_val_reg_epoch = -1

train_losses = []
train_mapes = []

val_losses = []
val_mapes = []

for epoch in range(num_epochs):
    train(model, train_loader, optimizer, criterion, device)

    train_loss = validate(model, train_loader, criterion, device)
    train_mape = validate_mape(model, train_loader)

    train_losses.append(train_loss)
    train_mapes.append(train_mape)

    scheduler.step(train_loss)

    val_loss = validate(model, val_loader, criterion, device)
    val_mape = validate_mape(model, val_loader)

    val_losses.append(val_loss)
    val_mapes.append(val_mape)

    val_mape_reg = validate_autoregressive_mape(model, val_loader, device)

    print(
        f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train MAPE: {train_mape:.4f}%, Validation Loss: {val_loss:.4f}, Validation MAPE: {val_mape:.4f}%, Val MAPE reg: {val_mape_reg:.4f}, lr: {scheduler.get_last_lr()}"
    )

    if train_mape < lowest_train_loss:
        lowest_train_loss = train_mape
        best_train_epoch = epoch + 1

    if val_mape < lowest_val_loss:
        lowest_val_loss = val_mape
        best_val_epoch = epoch + 1

    if val_mape_reg < lowest_val_reg_loss:
        lowest_val_reg_loss = val_mape_reg
        best_val_reg_epoch = epoch + 1

print(f"Lowest Train Loss: {lowest_train_loss:.6f} at Epoch {best_train_epoch}")
print(f"Lowest Val Loss: {lowest_val_loss:.6f} at Epoch {best_val_epoch}")
print(f"Lowest Val Reg Loss: {lowest_val_reg_loss:.6f} at Epoch {best_val_reg_epoch}")

# with log scale
plt.figure(figsize=(10, 6))
plt.plot(train_losses, linestyle="-", color="b", label="Training Loss")
plt.plot(val_losses, linestyle="-", color="r", label="Validation Loss")
plt.title("Loss Over Epochs (Log Scale)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale("log")
plt.grid(True)
plt.legend()
plt.savefig("plots/loss_log_scale.png")

# without log scale
plt.figure(figsize=(10, 6))
plt.plot(train_losses, linestyle="-", color="b", label="Training Loss")
plt.plot(val_losses, linestyle="-", color="r", label="Validation Loss")
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.savefig("plots/loss.png")

plt.figure(figsize=(10, 6))
plt.plot(train_mapes, linestyle="-", color="b", label="Training Loss")
plt.plot(val_mapes, linestyle="-", color="r", label="Validation Loss")
plt.title("MAPE Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MAPE (%)")
plt.grid(True)
plt.legend()
plt.savefig("plots/mape_loss.png")
