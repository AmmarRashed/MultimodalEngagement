import torch
import torch.nn as nn

from models.attention_rnn import AttentionRNN


class AttentiveSequentialClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1, output_size=5, dropout=0.1,
                 num_targets=1,
                 rnn_type='gru'):
        super(AttentiveSequentialClassifier, self).__init__()
        self.output_size = output_size

        self.rnn = AttentionRNN(input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                dropout=dropout,
                                rnn_type=rnn_type)

        self.fc_layers = nn.ModuleList()
        for target in range(num_targets):
            # init.xavier_uniform_(linear.weight)
            # init.constant_(linear.bias, 0)
            fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features=hidden_size, out_features=output_size))
            self.fc_layers.append(fc)

    def forward(self, x, lengths=None):
        # Process features using the GRU
        if lengths is not None:
            max_len = x.size(1)
            mask = torch.arange(max_len, device=x.device)[None, :] >= lengths[:, None]
        else:
            mask = None
        attention_out, rnn_out = self.rnn(x, src_key_padding_mask=mask)

        # Final classification
        return [fc(attention_out) for fc in self.fc_layers]
