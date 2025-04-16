import torch
import torch.nn as nn


class IRNN(nn.Module):
    def __init__(self, input_size, hidden_size, *args, **kwargs):
        super(IRNN, self).__init__()
        # Create an RNN layer with identity initialization and ReLU activation
        self.rnn = nn.RNN(input_size, hidden_size, nonlinearity='relu', bias=True,
                          *args, **kwargs)

        # Initialize RNN weights as identity matrix and biases to zero
        self.rnn.state_dict()['weight_hh_l0'].copy_(torch.eye(hidden_size))
        # Set the bias term to zero
        self.rnn.bias_ih_l0.data.fill_(0)
        self.rnn.bias_hh_l0.data.fill_(0)
        # Gaussian distribution with mean of zero and standard deviation of 0.001
        self.rnn.state_dict()['weight_ih_l0'].copy_(
            torch.randn(hidden_size, input_size) / 1000.0)

    def forward(self, x, *args, **kwargs):
        return self.rnn(x, *args, **kwargs)
