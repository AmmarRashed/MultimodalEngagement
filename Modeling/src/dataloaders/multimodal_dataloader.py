import os
from itertools import chain
from pathlib import Path
from typing import List, Tuple

import torch
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm.notebook import tqdm

from dataloaders.feature_extractors import *

FUNCTIONS = {
    "EEG": extract_eeg_features,
    "EYE": extract_eye_features,
    "HR": extract_hr_features,
    "OpenFace": extract_openface_features
}


def scale_sequences(sequences, scaler=MinMaxScaler(), fit_scaler=True):
    """
    Scale a list of sequences using sklearn's MinMaxScaler.
    Each sequence has shape (Ti, 70) where Ti is the sequence length.

    Parameters:
    -----------
    sequences : list of np.ndarray
        List of sequences to scale
    feature_range : tuple, optional (default=(0, 1))
        Desired range for scaled data

    Returns:
    --------
    scaled_sequences : list of np.ndarray
        Scaled sequences
    scaler : MinMaxScaler
        Fitted scaler for later use
    """

    # Fit the scaler on all sequences at once
    # Reshape to 2D array: (sum(Ti), 70)
    combined = np.vstack(sequences)
    if fit_scaler:
        scaler.fit(combined)

    # Transform each sequence
    scaled_sequences = [scaler.transform(seq) for seq in sequences]

    return scaled_sequences, scaler


class MultiModalDataset(Dataset):
    target_mappings = {t: i for i, t in enumerate(["engagement", "interest", "stress", "excitement"], start=2)}

    def __init__(self, root, modalities=["EEG", "EYE", "OpenFace"], targets=["engagement"],
                 participants=None,
                 class_mappings={0: 0, 1: 0, 2: 0, 3: 1, 4: 1},
                 scalers=None,
                 fit_scaler=True
                 ):
        root = Path(root)
        self.root = root
        self.modalities = modalities
        self.targets = targets
        if participants is None:
            participants = os.listdir(root)
        self.participants = self.get_valid_participants(participants)
        self.class_mappings = class_mappings
        if scalers is None:
            scalers = [MinMaxScaler() for _ in enumerate(self.modalities)]
        self.scalers = scalers
        self.fit_scaler = fit_scaler

        self.submissions = self.get_valid_submissions()
        self.X, self.y_raw, self.y_transformed = self._get_data()
        self.scaled_X, self.scalers = self.scale_data()

    def get_valid_participants(self, participants):
        """
        Filters out participants with a missing modality
        """
        valid = []
        for p in participants:
            p = str(p)
            is_valid = True
            for m in self.modalities:
                if not os.path.isdir(self.root / p / m):
                    is_valid = False
                    print(f"Participant {p} is missing at least one modality ({m})")
                    break
            if is_valid:
                valid.append(p)
        return valid

    def get_valid_submissions(self):
        """
        Filters out submissions with a missing modality
        """

        def step(p):
            submissions_per_modality = []
            for m in self.modalities:
                folder = self.root / p / m
                submissions_per_modality.append({f.stem for f in folder.glob("*.csv")})
            common_submissions = set.intersection(*submissions_per_modality)
            return list(common_submissions)
            # valid_submissions = []
            # for s in common_submissions:
            #     is_valid = True
            #     for m in self.modalities:
            #         df = pd.read_csv(self.root / p / m / f"{s}.csv")
            #         if len(df) < 2:
            #             is_valid = False
            #             break
            #     if is_valid:
            #         valid_submissions.append(s)
            # return valid_submissions

        submissions = Parallel(n_jobs=-1)(delayed(step)(p) for p in tqdm(self.participants, leave=False))
        return list(chain.from_iterable(submissions))

    def load_label(self, s):
        parts = s.split("_")
        submission_labels = []
        submission_labels_transformed = []
        for t in self.targets:
            l = int(parts[self.target_mappings[t]])
            submission_labels.append(l)

            l_transformed = self.class_mappings.get(l, None)
            submission_labels_transformed.append(l_transformed)
        return submission_labels, submission_labels_transformed

    def load_session(self, s):
        pid = s.split("_")[0]
        y_raw, y_transformed = self.load_label(s)
        sample = [[], y_raw, y_transformed]
        for m in self.modalities:
            path = self.root / pid / m / f"{s}.csv"
            df = pd.read_csv(path)
            if len(df) < 2:
                print(f"Insufficient {m} data for session {s}")
                return
            try:
                x = FUNCTIONS[m](df).dropna().values.astype('float32')
                if len(x) < 2:
                    print(f"Session {s} in {m} went bad")
                    return
            except Exception as e:
                print(f"Error loading {m} data for session {s}: {e}")
                return
            sample[0].append(x)
        return sample

    def _get_data(self):
        """
        list of feature vectors per modality
        """

        # data = [self.load_session(s) for s in
        #         tqdm(self.submissions, leave=False)]
        data = Parallel(n_jobs=-1)(delayed(self.load_session)(s)
                                   for s in tqdm(self.submissions, leave=False))
        data = [i for i in data if i is not None]

        X = [[] for _ in range(len(self.modalities))]
        y_raw = np.array([i[1] for i in data], dtype=int)
        y_transformed = np.array([i[2] for i in data], dtype=int)
        for x, _, _ in data:
            for i in range(len(self.modalities)):
                X[i].append(x[i])
        return X, y_raw, y_transformed

    def scale_data(self):
        scaled_X = []
        scalers = []
        for i, m in enumerate(self.modalities):
            scaled_data, scaler = scale_sequences(self.X[i], scaler=self.scalers[i], fit_scaler=self.fit_scaler)
            scaled_X.append(scaled_data)
            scalers.append(scaler)
        return scaled_X, scalers

    def __len__(self):
        return len(self.y_raw)

    def __getitem__(self, i):
        x = [torch.tensor(self.X[m][i], dtype=torch.float32) for m, _ in enumerate(self.modalities)]
        y_raw = self.y_raw[i]
        y_transformed = self.y_transformed[i]
        return tuple(x), torch.tensor(y_raw, dtype=torch.long), torch.tensor(y_transformed, dtype=torch.long)


class MultimodalCollator:
    """
    A collator function for multimodal datasets that handles padding of variable-length sequences.

    This collator expects each sample to contain:
    1. A tuple of tensors, where each tensor represents a modality with shape (Ti x Ki)
       where Ti is the sequence length and Ki is the feature dimension for modality i
    2. y_raw: target values with shape (1 x p) where p is number of target variables
    3. y_transformed: transformed target values with same shape as y_raw
    """

    def __init__(self, pad_value: float = 0.0):
        """
        Initialize the collator.

        Args:
            pad_value (float): Value to use for padding sequences. Defaults to 0.0.
        """
        self.pad_value = pad_value

    def __call__(self, batch: List[Tuple[Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]]) -> Tuple[
        Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        """
        Collate a batch of samples.

        Args:
            batch: List of tuples, where each tuple contains:
                  - Tuple of tensors for each modality
                  - y_raw tensor
                  - y_transformed tensor

        Returns:
            Tuple containing:
            - Tuple of padded tensors for each modality (N x max_Ti x Ki)
            - Batched y_raw (N x p)
            - Batched y_transformed (N x p)
        """
        # Split the batch into components
        modality_samples, y_raw, y_transformed = zip(*batch)

        # Number of modalities
        num_modalities = len(modality_samples[0])

        # Process each modality separately
        padded_modalities = []
        for modality_idx in range(num_modalities):
            # Extract sequences for current modality
            modality_sequences = [sample[modality_idx] for sample in
                                  modality_samples]

            # Pad sequences to longest sequence in this modality
            padded_modality = pad_sequence(
                modality_sequences,
                batch_first=True,
                padding_value=self.pad_value
            )
            padded_modalities.append(padded_modality)

        # Stack the targets
        y_raw_batched = torch.stack(y_raw)
        y_transformed_batched = torch.stack(y_transformed)

        return tuple(padded_modalities), y_raw_batched, y_transformed_batched
