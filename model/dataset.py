import numpy as np
import torch

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class ImpurityDataset(Dataset):
    def __init__(self, dataframe, feature_names, label_names, feature_scaler=None, label_scaler=None, device=None):
        assert len(label_names) % 2 == 0, "Labels must occur in pairs of real and imaginary numbers."

        self.n_samples = len(dataframe)
        self.encoder_feature_names = feature_names
        self.label_names = label_names

        self.device = device

        feature_data, label_data = self._prepare_data(dataframe, feature_scaler, label_scaler)

        self.encoder_input = torch.tensor(feature_data, dtype=torch.float).to(device)
        self.label_data = torch.tensor(label_data, dtype=torch.float).to(device)

        N, _, C = self.label_data.shape
        start_token = torch.zeros((N, 1, C), dtype=self.label_data.dtype).to(device)
        self.decoder_input = torch.cat((start_token, self.label_data[:, :-1, :]), dim=1)

    def _prepare_data(self, dataframe, feature_scaler, label_scaler):
        df_encoder_input = dataframe[self.encoder_feature_names]
        df_labels = dataframe[self.label_names]

        xs = feature_scaler.transform(df_encoder_input.values) if feature_scaler else df_encoder_input.values
        ys = label_scaler.transform(df_labels.values) if label_scaler else df_labels.values

        N, S, C = self.n_samples, len(self.encoder_feature_names), 1
        xs = xs.reshape(N, S, C)
        features = np.zeros((N, S, C))  # endoder input

        N, T, C = self.n_samples, len(self.label_names) // 2, 2
        ys = ys.reshape(N, T, C)
        labels = np.zeros((N, T, C))  # decoder output

        for i in range(self.n_samples):
            features[i] = xs[i]
            labels[i] = ys[i]

        return features, labels

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.encoder_input[idx], self.decoder_input[idx], self.label_data[idx]


def compute_scalers(dataframe, fixed_features, labels, test_size, seed):
    assert seed != None, "Not defining a seed can cause unexpected behaviour, make sure that this is intended."

    train_df, _ = train_test_split(dataframe, test_size=test_size, random_state=seed)
    df_features = train_df[fixed_features]
    df_labels = train_df[labels]

    feature_scaler = StandardScaler().fit(df_features.values)
    label_scaler = StandardScaler().fit(df_labels.values)

    return feature_scaler, label_scaler
