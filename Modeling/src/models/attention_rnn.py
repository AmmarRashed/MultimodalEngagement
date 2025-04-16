import torch
import torch.nn as nn
import torch.nn.functional as F
from models.irnn import IRNN


class AttentionRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0,
                 rnn_type='gru'):
        super(AttentionRNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn_type = rnn_type
        # RNN
        kwargs = dict(hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        if rnn_type == 'gru':
            self.rnn = nn.GRU(input_size, **kwargs)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, **kwargs)
        elif rnn_type == 'irnn':
            self.rnn = IRNN(input_size, **kwargs)
        else:
            raise Exception("Unknown RNN")

        # Attention layer
        self.attention_linear = nn.Linear(hidden_size, hidden_size)
        self.attention_combine = nn.Linear(hidden_size * 2, hidden_size)

        self.dropout = nn.Dropout(dropout)

    def attention(self, hidden, encoder_outputs):
        energy = torch.tanh(self.attention_linear(encoder_outputs))
        attention_scores = F.softmax(torch.bmm(energy, hidden.unsqueeze(2)).squeeze(2), dim=1)
        attention_scores = self.dropout(attention_scores)
        context = torch.bmm(attention_scores.unsqueeze(1), encoder_outputs).squeeze(1)
        combined = torch.cat((hidden, context), dim=1)
        output = torch.tanh(self.attention_combine(combined))
        return output

    def forward(self, input_seq):
        batch_size, seq_len, _ = input_seq.size()

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(input_seq.device)
        if self.rnn_type == 'lstm':
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(input_seq.device)
            out, (hn, cn) = self.rnn(input_seq, (h0, c0))
        else:
            # LSTM input shape: (batch_size, seq_len, input_size)
            out, hn = self.rnn(input_seq, h0)

        # Attention mechanism
        attention_out = self.attention(hn[-1], out)

        return attention_out, out