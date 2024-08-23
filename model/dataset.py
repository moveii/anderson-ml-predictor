import numpy as np
import torch

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class ImpurityDataset(Dataset):
    def __init__(self, dataframe, encoder_input, labels, pair_up=False, use_scaling=False, test_size=0.1, device=None, seed=None):
        self.n_samples = len(dataframe)
        self.encoder_feature_names = encoder_input
        self.label_names = labels

        self.device = device

        self.feature_scaler, self.label_scaler = (
            self._compute_scalers(dataframe, pair_up, test_size, seed) if use_scaling else (None, None)
        )

        feature_data, label_data = self._prepare_data(dataframe, pair_up)

        self.encoder_input = torch.tensor(feature_data, dtype=torch.float).to(device)
        self.label_data = torch.tensor(label_data, dtype=torch.float).to(device)

        N, _, C = self.label_data.shape
        start_token = torch.zeros((N, 1, C), dtype=self.label_data.dtype).to(device)
        self.decoder_input = torch.cat((start_token, self.label_data[:, :-1, :]), dim=1)

    def _compute_scalers(self, dataframe, pair_up, test_size, seed):
        assert seed != None, "Not defining a seed can cause unexpected behaviour, make sure that this is intended."

        train_df, _ = train_test_split(dataframe, test_size=test_size, random_state=seed)
        df_features = train_df[self.encoder_feature_names]
        df_labels = train_df[self.label_names]

        if pair_up:
            # we order the labels in pairs (1 and 122, 2 and 121, and so on if pair_size == 2)
            df_labels = self._reorder_columns(df_labels)

        feature_scaler = StandardScaler().fit(df_features.values)
        print(df_labels.values[0])
        label_scaler = StandardScaler().fit(df_labels.values)

        return feature_scaler, label_scaler

    def _prepare_data(self, dataframe, pair_up):
        pair_size = 2 if pair_up else 1
        assert len(self.label_names) % pair_size == 0, "Labels must occur in pairs."

        df_encoder_input = dataframe[self.encoder_feature_names]
        df_labels = dataframe[self.label_names]

        if pair_up:
            # we order the labels in pairs (1 and 122, 2 and 121, and so on if pair_size == 2)
            df_labels = self._reorder_columns(df_labels)

        xs = self.feature_scaler.transform(df_encoder_input.values) if self.feature_scaler else df_encoder_input.values
        ys = self.label_scaler.transform(df_labels.values) if self.label_scaler else df_labels.values

        N, S, C = self.n_samples, len(self.encoder_feature_names), 1
        xs = xs.reshape(N, S, C)
        features = np.zeros((N, S, C))  # endoder input

        N, T, C = self.n_samples, len(self.label_names) // pair_size, pair_size
        ys = ys.reshape(N, T, C)
        labels = np.zeros((N, T, C))  # decoder output

        for i in range(self.n_samples):
            features[i] = xs[i]
            labels[i] = ys[i]

        return features, labels

    def _reorder_columns(self, df):
        cols = df.columns.tolist()
        new_order = []

        for i in range(len(cols) // 2):
            new_order.append(cols[i])
            new_order.append(cols[-(i + 1)])

        return df[new_order]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.encoder_input[idx], self.decoder_input[idx], self.label_data[idx]

    def __get_label_scaler__(self):
        return self.label_scaler

    def __get_feature_scaler__(self):
        return self.feature_scaler
