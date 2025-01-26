import os
import random
import warnings
from typing import Union

import numpy as np
import torch

from dataloaders.features_dataset import FeaturesDataset


class WebcamFeaturesDataset(FeaturesDataset):
    datasets = None  # placeholder for combining multiple modalities

    def __init__(self, root_dir, selected_participants=None, selected_sessions=None,
                 max_frames=None, sampling_rate: Union[dict, int] = 3,
                 labels_mapper=None,
                 selected_features=["expression", "OpenFace_pose", "OpenFace_gaze", "OpenFace_blink"],
                 scale_by=(0, 1), targets=["engagement"],
                 unablated_feature_indices=None
                 ):
        """
        Args:
            root_dir:
            selected_participants:
            max_frames:
            sampling_rate:
            labels_mapper:
            selected_features:
                available options:
                    "mirror" to include mirrored samples
                    "shift" to include shifted frames from the sequences
                    The boolean if True uses class weights in augmentation
            scale_by: a tuple of mean and std to scale features with. Default (0 mean, 1 std)
            targets: a list of targets to include.
                Available options are {'engagement', 'interest', 'stress', 'excitement'}
        """

        self.selected_participants = selected_participants
        self.selected_sessions = selected_sessions
        self.max_frames = max_frames
        if isinstance(sampling_rate, int):
            sampling_rate = {f: sampling_rate for f in selected_features}
        else:
            self.sampling_rate = sampling_rate
        self.sampling_rate = sampling_rate
        self.labels_mapper = labels_mapper
        self.selected_features = selected_features
        self.mirrored_data_dir = None
        self.augmentations = None
        self.set_scaler(*scale_by)  # mean & std
        self.targets = targets
        self.unablated_feature_indices = unablated_feature_indices
        super(WebcamFeaturesDataset, self).__init__(root_dir)

        if max_frames is None:
            print("Max Frames not specified")
            max_frames = self._infer_max_frames()
        self.max_frames = max_frames

        self.labels = self.get_all_labels()
        if self.labels_mapper is not None:
            self.num_classes = len(set(self.labels_mapper.values()))
        else:
            self.num_classes = len(set([l[0] for l in self.labels if l is not None]))

    def _infer_max_frames(self):
        max_len = 0
        for path, _, _ in self.data:
            if not self.valid_x(path):
                continue
            file = np.load(path)
            for f in self.selected_features:
                feature_len = len(file[f][::self.sampling_rate[f]])
                max_len = max(max_len, feature_len)
        return max_len

    def set_scaler(self, mean, std):
        self.scale_by = (mean, std)

    def set_mirrored_data_dir(self, folder):
        if folder:
            self.mirrored_data_dir = folder

    def augment_data(self, augmentations=[("mirror", True)]):
        """
        Args:
            augmentations: a list of tuples (strategy name, boolean for balancing the dataset)
        Returns:

        """
        self.augmentations = augmentations
        if augmentations is None:
            augmentations = list()
        for (aug, balance) in augmentations:
            if aug.lower().strip() == "mirror":
                self.augment_with_mirrors(balance)
            elif aug.lower().strip() == "shift":
                self.augment_with_shifts(balance)

    def _augment_data(self, class_to_extra_samples, class_freq, balance=True):
        if balance:
            # 1 add all mirrored samples for least represented class
            # 2 add mirrored samples for other classes without exceeding the number of samples for the min class

            min_class = np.argmin(class_freq)
            min_class_count = len(class_to_extra_samples[min_class]) + class_freq[min_class]
            # print(f"Min Class {min_class} Count:", min_class_count)
            extra = list()
            for samples in class_to_extra_samples:
                # print(f"{c} has", len(mirrored))
                diff = min_class_count - len(samples)
                diff = max(diff, 0)
                diff = min(diff, len(samples))
                # print(f"So it needs: {diff}")
                # notice that the biggest diff is when len(mirrored) is smallest, which is min_class_count
                extra.extend(random.sample(samples, diff))
        else:
            extra = list()
            for samples in class_to_extra_samples:
                extra.extend(samples)
        self.data.extend(extra)

    def get_balanced_subset(self):
        class_freq = np.zeros(self.num_classes, dtype=int)
        indices = {i: [] for i in range(self.num_classes)}
        for i, (video_path, targets, shift) in enumerate(self.data):
            label = targets[0]
            if label == -1:
                continue
            class_freq[label] += 1
            indices.setdefault(label, [])
            indices[label].append(i)

        min_class_count = np.min(class_freq)
        balanced_subset = []
        for c, class_indices in indices.items():
            balanced_subset.extend(random.sample(class_indices, min_class_count))
        return balanced_subset

    def augment_with_mirrors(self, balance=True):
        # samples per class
        class_to_extra_samples = [[] for _ in range(self.num_classes)]
        class_freq = np.zeros(self.num_classes, dtype=int)
        for video_path, targets, shift in self.data:
            label = targets[0]
            if label == -1:
                continue
            class_freq[label] += 1
            filename = os.path.basename(video_path)
            pid = filename.split('_')[0]
            class_to_extra_samples[label].append((os.path.join(self.mirrored_data_dir, pid, filename), targets, shift))

        self._augment_data(class_to_extra_samples, class_freq, balance)

    def augment_with_shifts(self, balance=True):
        sr_values = set(self.sampling_rate.values())
        if len(sr_values) > 1:
            warnings.warn("Multiple sampling rates detected. Skipping shift augmentation")
            return
        sampling_rate = sr_values.pop()
        if sampling_rate == 1:
            return
        class_to_extra_samples = [[] for _ in range(self.num_classes)]
        class_freq = np.zeros(self.num_classes, dtype=int)
        for video_path, targets, _ in self.data:
            label = targets[0]
            if label == -1:
                continue
            class_freq[label] += 1

            for shift in range(1, sampling_rate):
                class_to_extra_samples[label].append((video_path, targets, shift))

        self._augment_data(class_to_extra_samples, class_freq, balance)

    def parse_path(self, path):
        def data_split_filter(value, selections):
            return selections is None or value in selections

        filename = os.path.basename(path)
        participant_id, session_id, engagement, interest, stress, excitement = filename.split('.')[0].split('_')
        y = []
        for target in self.targets:
            l = int(eval(target))
            if self.labels_mapper is not None:
                l = self.labels_mapper.get(l, -1)
            y.append(l)
        y = np.array(y)
        take_participant = data_split_filter(participant_id, self.selected_participants)
        take_session = data_split_filter(session_id, self.selected_sessions)
        if not (take_participant and take_session):
            return
        return path, y, 0

    def parse_data(self):
        data = []
        for folder, _, files in os.walk(self.root_dir):
            for f in files:
                if f.startswith('.'):
                    continue
                result = self.parse_path(os.path.join(folder, f))
                if result is None or result[1][0] is None:
                    continue
                if self.valid_x(result[0]):
                    data.append(result)
        return data

    def get_all_labels(self):
        return np.array([label for _, label, _ in self.data])

    def valid_x(self, path, selected_features=None):
        if selected_features is None:
            selected_features = self.selected_features
        file = np.load(path)
        for f in selected_features:
            if len(file[f]) < 2:
                return False
        return True

    @staticmethod
    def _load_x(path, selected_features, shift=0, sampling_rate=3, max_frames=600, unablated_feature_indices=None):
        if isinstance(sampling_rate, int):
            sampling_rate = {f: sampling_rate for f in selected_features}
        file = np.load(path)
        features = []
        for f in selected_features:
            feature = file[f][shift::sampling_rate[f]]
            features.append(feature)
            if any([word in f for word in ["pose", "gaze"]]):
                velocity = np.absolute(np.gradient(feature, axis=0))
                acceleration = np.absolute(np.gradient(velocity, axis=0))
                features.append(velocity)
                features.append(acceleration)
        min_len = min([len(x) for x in features])
        x = [x[:min_len] for x in features]
        x = torch.tensor(np.concatenate(x, axis=-1)).to(torch.float32)
        # x = x[shift::self.sampling_rate]
        if len(x) > max_frames:
            x = x[-max_frames:]
        if unablated_feature_indices is None:
            return x
        return x[:, unablated_feature_indices]

    def load_x(self, path, shift=0, selected_features=None):
        if selected_features is None:
            selected_features = self.selected_features
        return self._load_x(path, selected_features, shift, self.sampling_rate, self.max_frames,
                            self.unablated_feature_indices)

    def compute_features_mean_std(self):
        features = torch.cat([self.__getitem__(i)[0] for i in range(len(self.data))])
        mean = features.mean(dim=0)
        std = features.std(dim=0)
        return mean, std

    def __getitem__(self, index):
        if self.datasets is not None:
            return self.combine_datasets(index)

        tensor_path, label, shift = self.data[index]
        x = self.load_x(tensor_path, shift)
        mean, std = self.scale_by
        x = (x - mean) / std
        return x, torch.tensor(label)

    def combine_datasets(self, index):
        xs = []
        for d in self.datasets:
            x, y = d.__getitem__(index)
            xs.append(x)
        xs = torch.cat(xs, dim=1)
        return xs, y


def collate_fn(batch):
    # Unpack the batch
    features, labels = zip(*batch)

    # Get lengths before padding
    lengths = torch.tensor([len(seq) for seq in features])

    # Pad sequences to max length in this batch
    max_len = max(lengths)
    feature_dim = features[0].shape[-1]
    padded_features = torch.zeros(len(batch), max_len, feature_dim)

    for i, seq in enumerate(features):
        padded_features[i, :len(seq)] = seq

    labels = torch.stack(labels)

    return padded_features, labels, lengths
