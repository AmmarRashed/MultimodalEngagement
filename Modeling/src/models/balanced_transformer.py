import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BalancedTransformerClassifier(nn.Module):
    def __init__(self, input_size, output_size,
                 hidden_size=64, num_layers=1, nheads=1, dropout=0.1,
                 num_targets=1,
                 project_to=None):
        super().__init__()

        self.project_to = project_to
        if project_to is not None:
            self.projection = nn.Sequential(
                nn.Linear(input_size, project_to),
                nn.LayerNorm(project_to),
                nn.Dropout(dropout)
            )
            input_size = project_to

        # Positional encoding for better temporal modeling
        self.pos_encoder = PositionalEncoding(input_size, dropout)

        # Main transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            batch_first=True,
            dim_feedforward=hidden_size,
            nhead=nheads,
            dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Two-stage classification head
        self.classifiers = nn.ModuleList()
        for target in range(num_targets):
            classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, output_size)
            )
            self.classifiers.append(classifier)

    def forward(self, x, lengths=None):
        # Handle variable length sequences with masking
        if lengths is not None:
            max_len = x.size(1)
            mask = torch.arange(max_len, device=x.device)[None, :] >= lengths[:, None]
        else:
            mask = None

        # Initial projection if needed
        if self.project_to is not None:
            x = self.projection(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Encode sequence
        x = self.encoder(x, src_key_padding_mask=mask)

        # Masked pooling for variable length sequences
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)
            x = x.sum(dim=1) / (~mask).sum(dim=1, keepdim=True)
        else:
            x = x.mean(dim=1)

        # Classification
        return torch.stack([clf(x) for clf in self.classifiers])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class OrdinalBCELoss(nn.Module):
    def __init__(self, alpha=0.5, threshold=2, device='cuda', pos_weight=None, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.threshold = threshold
        self.device = device
        self.pos_weight = pos_weight
        self.reduction = reduction

        # Pre-compute binary targets for efficiency
        self.binary_targets = torch.tensor([1 if i > threshold else 0
                                            for i in range(5)]).float().to(device)

    def forward(self, logits, targets):
        # Convert ordinal targets to binary
        binary_targets = self.binary_targets[targets.long()]

        # BCE Loss
        bce = F.binary_cross_entropy_with_logits(
            logits,
            binary_targets,
            reduction='none',
            pos_weight=self.pos_weight
        )

        # Ordinal penalty
        normalized_targets = targets.float() / 4.0  # Scale to [0,1]
        predicted_probs = torch.sigmoid(logits)
        ordinal_penalty = F.mse_loss(
            predicted_probs,
            normalized_targets,
            reduction='none'
        )

        loss = bce + self.alpha * ordinal_penalty
        return loss.mean() if self.reduction == 'mean' else loss.sum() if self.reduction == 'sum' else loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def get_balanced_sampler(dataset):
    """
    Creates a balanced batch sampler that ensures equal representation
    of both classes in each batch
    """
    labels = torch.tensor([label for _, label in dataset])
    class_sample_count = torch.tensor(
        [(labels == t).sum() for t in torch.unique(labels, sorted=True)]
    )
    weight = 1. / class_sample_count.float()
    samples_weight = torch.tensor([weight[t] for t in labels])

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=samples_weight,
        num_samples=len(samples_weight),
        replacement=True
    )

    return sampler
