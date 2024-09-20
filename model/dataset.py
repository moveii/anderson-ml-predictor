# Patrick Berger, 2024

import numpy as np
import torch
import pandas as pd
from typing import List, Optional, Tuple
import logging
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class ImpurityDataset(Dataset):
    PAIR_SIZE = 2

    def __init__(
        self,
        dataframe: pd.DataFrame,
        encoder_input: List[str],
        labels: List[str],
        pair_up_labels: bool = False,
        use_scaling: bool = False,
        validation_size: float = 0.1,
        test_size: float = 0.1,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the ImpurityDataset.

        Args:
            dataframe (pd.DataFrame): Input dataframe
            encoder_input (List[str]): List of feature column names
            labels (List[str]): List of label column names
            pair_up_labels (bool): Whether to pair up labels, so that t(i) and t(max-i) are predicted at the same time
            use_scaling (bool): Whether to use feature and label scaling
            validation_size (float): Proportion of the dataset to include in the validation split
            test_size (float): Proportion of the dataset to include in the test split
            device (torch.device, optional): Device to store tensors on
            seed (int, optional): Random seed for train-test split
        """
        self.n_samples = len(dataframe)
        self.encoder_feature_names = encoder_input
        self.label_names = labels
        self.device = device or torch.device("cpu")
        self.seed = seed

        if self.seed is None:
            logging.warning("No seed provided. This may cause unexpected behavior.")

        indices = list(range(self.n_samples))
        other_size = validation_size + test_size
        train_indices, other_indices = train_test_split(indices, test_size=other_size, random_state=self.seed)

        val_size = validation_size / other_size
        val_indices, test_indices = train_test_split(other_indices, test_size=(1 - val_size), random_state=self.seed)

        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices

        self._feature_scaler, self._label_scaler = (
            self._compute_scalers(dataframe.iloc[train_indices], pair_up_labels) if use_scaling else (None, None)
        )

        feature_data, label_data = self._prepare_data(dataframe, pair_up_labels)

        self.encoder_input = torch.tensor(feature_data, dtype=torch.float).to(self.device)
        self.label_data = torch.tensor(label_data, dtype=torch.float).to(self.device)

        N, _, C = self.label_data.shape
        start_token = torch.zeros((N, 1, C), dtype=self.label_data.dtype).to(self.device)
        self.decoder_input = torch.cat((start_token, self.label_data[:, :-1, :]), dim=1)

    def _compute_scalers(self, train_df: pd.DataFrame, pair_up_labels: bool) -> Tuple[StandardScaler, StandardScaler]:
        """
        Compute feature and label scalers.

        Args:
            train_df (pd.DataFrame): Training dataframe
            pair_up (bool): Whether to pair up labels

        Returns:
            Tuple[StandardScaler, StandardScaler]: Feature and label scalers
        """
        df_features = train_df[self.encoder_feature_names]
        df_labels = train_df[self.label_names]

        if pair_up_labels:
            df_labels = self._reorder_labels(df_labels)

        feature_scaler = StandardScaler().fit(df_features.values)
        label_scaler = StandardScaler().fit(df_labels.values)

        return feature_scaler, label_scaler

    def _prepare_data(self, dataframe: pd.DataFrame, pair_up_labels: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare feature and label data.

        Args:
            dataframe (pd.DataFrame): Input dataframe
            pair_up_labels (bool): Whether to pair up labels

        Returns:
            Tuple[np.ndarray, np.ndarray]: Prepared feature and label data
        """
        pair_size = self.PAIR_SIZE if pair_up_labels else 1
        if len(self.label_names) % pair_size != 0:
            raise ValueError("Number of labels must be divisible by pair size.")

        df_encoder_input = dataframe[self.encoder_feature_names]
        df_labels = dataframe[self.label_names]

        if pair_up_labels:
            df_labels = self._reorder_labels(df_labels)

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

    def _reorder_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reorder labels for pairing (time step 1 and 122 are paired, 2 and 121, and so on).

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: Dataframe with reordered columns
        """
        cols = df.columns.tolist()
        if len(cols) % 2 != 0:
            raise ValueError("Number of columns must be even for pairing.")

        new_order = []
        for i in range(len(cols) // 2):
            new_order.append(cols[i])
            new_order.append(cols[-(i + 1)])

        return df[new_order]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.encoder_input[idx], self.decoder_input[idx], self.label_data[idx]

    @property
    def label_scaler(self) -> Optional[StandardScaler]:
        return self._label_scaler

    @property
    def feature_scaler(self) -> Optional[StandardScaler]:
        return self._feature_scaler

    def get_train_dataset(self) -> Subset:
        return Subset(self, self.train_indices)

    def get_val_dataset(self) -> Subset:
        return Subset(self, self.val_indices)

    def get_test_dataset(self) -> Subset:
        return Subset(self, self.test_indices)
