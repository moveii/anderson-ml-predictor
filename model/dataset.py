import random
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class ImpurityDataset(Dataset):
    def __init__(self, dataframe, encoder_features, labels, feature_scaler=None, label_scaler=None, device=None):
        assert len(labels) % 2 == 0, "Labels must occur in pairs of real and imaginary numbers."

        self.n_samples = len(dataframe)
        self.encoder_features = encoder_features
        self.labels = labels
        self.device = device

        self.feature_data, self.label_data = self._prepare_data(dataframe, feature_scaler, label_scaler)

        self.encoder_features = torch.tensor(self.feature_data, dtype=torch.float).to(device)
        self.labels = torch.tensor(self.label_data, dtype=torch.float).to(device)

    def _prepare_data(self, dataframe, feature_scaler, label_scaler):
        df_encoder_features = dataframe[self.encoder_features]
        df_labels = dataframe[self.labels]

        xs = feature_scaler.transform(df_encoder_features.values) if feature_scaler else df_encoder_features.values
        ys = label_scaler.transform(df_labels.values) if label_scaler else df_labels.values

        N, S, C = self.n_samples, len(self.encoder_features), 1
        xs = xs.reshape(N, S, C)
        features = np.zeros((N, S, C))  # endoder input

        N, T, C = self.n_samples, len(self.labels) // 2, 2
        ys = ys.reshape(N, T, C)
        labels = np.zeros((N, T, C))  # decoder output

        for i in range(self.n_samples):
            features[i] = xs[i]
            labels[i] = ys[i]

        return features, labels

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.encoder_features[idx], self.labels[idx]


def compute_scalers(dataframe, fixed_features, labels, test_size, seed):
    assert seed != None, "Not defining a seed can cause unexpected behaviour, make sure that this is intended."

    train_df, _ = train_test_split(dataframe, test_size=test_size, random_state=seed)
    df_features = train_df[fixed_features]
    df_labels = train_df[labels]

    feature_scaler = StandardScaler().fit(df_features.values)
    label_scaler = StandardScaler().fit(df_labels.values)

    return feature_scaler, label_scaler
