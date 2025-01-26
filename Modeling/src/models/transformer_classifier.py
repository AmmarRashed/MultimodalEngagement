import torch
import torch.nn as nn


class TransformerClassifier(nn.Module):
    def __init__(self, input_size, output_size,
                 hidden_size=64, num_layers=1, nheads=1, dropout=0.1,
                 num_targets=1,
                 project_to=None
                 ):
        """
        Args:
            input_size: Embedding size of the frame embedding model
            output_size: Number of Classes
            hidden_size: the dimension of the transformer's feedforward network model
            num_layers: number of encoder layers
            nheads: number of attention heads
            dropout: dropout rate
        """

        super(TransformerClassifier, self).__init__()
        self.project_to = project_to
        if project_to is not None:
            self.projection = nn.Sequential(nn.Linear(input_size, project_to), nn.Dropout(dropout))
            input_size = project_to
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, batch_first=True,
                                                        dim_feedforward=hidden_size,
                                                        nhead=nheads,
                                                        dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc_layers = nn.ModuleList()
        for target in range(num_targets):
            fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features=input_size, out_features=output_size))
            self.fc_layers.append(fc)

    def forward(self, x, lengths=None):
        if lengths is not None:
            max_len = x.size(1)
            mask = torch.arange(max_len, device=x.device)[None, :] >= lengths[:, None]
        else:
            mask = None

        # Transformer Encoder
        if self.project_to is not None:
            x = self.projection(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        x = x.mean(dim=1)

        # Classification layer
        return torch.stack([fc(x) for fc in self.fc_layers])

    def forward_single_target(self, x):
        return self.forward(x)[0]
