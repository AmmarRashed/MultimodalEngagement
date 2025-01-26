from abc import ABC, abstractmethod

import numpy as np
from sklearn.utils import compute_class_weight
from torch.utils.data import Dataset


class FeaturesDataset(ABC, Dataset):
    def __init__(self, root_dir, one_hot=False):
        self.root_dir = root_dir
        self.one_hot = one_hot

        self.data = self.parse_data()

    @abstractmethod
    def parse_data(self):
        raise NotImplementedError

    @abstractmethod
    def get_all_labels(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def get_class_weights(self, y=None):
        if y is None:
            y = self.get_all_labels()
        if len(y.shape) > 1:
            return [self.get_class_weights(y[y[:, i] != -1, i]) for i in range(y.shape[1])]

        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        return class_weights

    def compute_features_mean_std(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError
