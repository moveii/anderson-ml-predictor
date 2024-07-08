import random
import numpy as np
import torch
import pandas as pd
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


df = pd.read_csv(file_path, skiprows=4)   # we skip the first four lines, because they are just metadata
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

print(dataset.__getitem__(0))