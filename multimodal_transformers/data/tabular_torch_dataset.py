from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset as TorchDataset


class TorchTabularTextDataset(TorchDataset):
    """
    :obj:`TorchDataset` wrapper for text dataset with categorical features
    and numerical features

    :param encodings:
        The output from `encode_plus()` and `batch_encode()` methods (tokens, attention_masks, etc.) of a `transformers.PreTrainedTokenizer`.

    :param categorical_feats:
        An array containing the preprocessed categorical features. Shape: `(n_examples, categorical feat dim)`.

    :param numerical_feats:
        An array containing the preprocessed numerical features. Shape: `(n_examples, numerical feat dim)`.

    :param labels:
        The labels of the training examples.

    :param df:
        The original dataset. Optional and used only to save the original dataset with the preprocessed dataset.

    :param label_list:
        A list of class names for each unique class in labels.
    """

    def __init__(
        self,
        encodings: transformers.BatchEncoding,
        # categorical_feats: Optional[pd.DataFrame],
        categorical_feats: Optional[np.ndarray],
        numerical_feats: Optional[np.ndarray],
        numerical_labels: Optional[np.ndarray],
        labels: Optional[Union[List, np.ndarray]] = None,
        df: Optional[pd.DataFrame] = None,
        label_list: Optional[List[Union[str]]] = None,
        cat_mask: Optional[np.ndarray] = None,
        numerical_mask: Optional[np.ndarray] = None,
    ):
        self.df = df
        self.encodings = encodings
        # self.cat_feats = categorical_feats.values if categorical_feats is not None else None
        self.cat_feats = categorical_feats if categorical_feats is not None else None
        self.numerical_feats = numerical_feats
        self.numerical_labels = numerical_labels
        self.labels = labels
        self.label_list = (
            label_list
            if label_list is not None
            else [i for i in range(len(np.unique(labels)))]
        )
        self.cat_mask = cat_mask
        self.numerical_mask = numerical_mask

    def __getitem__(self, idx: int):
        item = {}
        item["labels"] = (
            torch.tensor(self.labels[idx]) if self.labels is not None else None
        )
        item["cat_feats"] = (
            torch.tensor(self.cat_feats[idx])
            if self.cat_feats is not None
            else torch.zeros(0)
        )
        item["numerical_feats"] = (
            torch.tensor(self.numerical_feats[idx]).float()
            if self.numerical_feats is not None
            else torch.zeros(0)
        )
        item["numerical_labels"] = (
            torch.tensor(self.numerical_labels[idx]).float()
            if self.numerical_labels is not None
            else torch.zeros(0)
        )
        item["cat_mask"] = (
            torch.tensor(self.cat_mask[idx]).bool()
            if self.cat_mask is not None else None
        )
        item["numerical_mask"] = (
            torch.tensor(self.numerical_mask[idx]).bool()
            if self.numerical_mask is not None else None
        )
        return item

    def __len__(self) -> int:
        return len(self.numerical_feats)

    def get_labels(self) -> Optional[List[Union[str]]]:
        """Returns the label names for classification."""
        return self.label_list
