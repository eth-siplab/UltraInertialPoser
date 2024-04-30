r"""
    Utils for RNN including networks, datasets, and loss wrappers.
"""


__all__ = ['RNN', 'RNNWithInit',"MLP"]


import os
import torch.utils.data
from torch.nn.functional import relu
from torch.nn.utils.rnn import *
from torch import nn as nn
class LinearLayers(nn.Module):
    """
    One or multiple dense layers with skip connections from input to final output.
    """

    def __init__(self, hidden_size, num_layers=2, dropout_p=0.0, use_skip=False, use_batch_norm=True):
        super(LinearLayers, self).__init__()
        self.hidden_size = hidden_size

        layers = []
        for _ in range(num_layers):
            new_layers = [nn.Linear(hidden_size, hidden_size)]
            if use_batch_norm:
                bn = nn.BatchNorm1d(hidden_size)
                nn.init.uniform_(bn.weight)
                new_layers.append(bn)
            new_layers.append(nn.PReLU())
            new_layers.append(nn.Dropout(dropout_p))
            layers.extend(new_layers)

        self.layers = nn.Sequential(*layers)

        if use_skip:
            self.skip = lambda x, y: x + y
        else:
            self.skip = lambda x, y: y

    def forward(self, x):
        y = self.layers(x)
        out = self.skip(x, y)
        return out


class MLP(nn.Module):
    """
    An MLP mapping from input size to output size going through n hidden dense layers. Uses batch normalization,
    PReLU and can be configured to apply dropout.
    """

    def __init__(self, input_size, output_size, hidden_size, num_layers=2, dropout_p=0.0, skip_connection=False,
                 use_batch_norm=True):
        super(MLP, self).__init__()
        self.input_to_hidden = nn.Linear(input_size, hidden_size)
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(hidden_size)
            nn.init.uniform_(self.batch_norm.weight)
        else:
            self.batch_norm = nn.Identity()
        self.activation_fn = nn.PReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.hidden_to_output = nn.Linear(hidden_size, output_size)
        hidden_layers = []
        for _ in range(num_layers):
            h = LinearLayers(hidden_size, dropout_p=dropout_p, use_batch_norm=use_batch_norm, use_skip=skip_connection)
            hidden_layers.append(h)
        self.hidden_layers = nn.Sequential(*hidden_layers)

    def forward(self, x):
        y = self.input_to_hidden(x)
        y = self.batch_norm(y)
        y = self.activation_fn(y)
        y = self.dropout(y)
        y = self.hidden_layers(y)
        y = self.hidden_to_output(y)
        return y

class RNN(torch.nn.Module):
    r"""
    An RNN net including a linear input layer, an RNN, and a linear output layer.
    """
    def __init__(self, input_size: int, output_size: int, hidden_size: int, num_rnn_layer: int,
                 rnn_type='lstm', bidirectional=False, dropout=0., load_weight_file: str = None):
        r"""
        Init an RNN.

        :param input_size: Input size.
        :param output_size: Output size.
        :param hidden_size: Hidden size for RNN.
        :param num_rnn_layer: Number of RNN layers.
        :param rnn_type: Select from 'rnn', 'lstm', 'gru'.
        :param bidirectional: Whether if the RNN is bidirectional.
        :param dropout: Dropout after the input linear layer and in the rnn.
        :param load_weight_file: If not None and exists, weights will be loaded.
        """
        super().__init__()
        self.rnn = getattr(torch.nn, rnn_type.upper())(hidden_size, hidden_size, num_rnn_layer,
                                                       bidirectional=bidirectional, dropout=dropout)
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()

        if load_weight_file and os.path.exists(load_weight_file):
            self.load_state_dict(torch.load(load_weight_file))
            self.eval()

    def forward(self, x, init=None):
        r"""
        Forward.

        :param x: A list in length [batch_size] which contains tensors in shape [num_frames, input_size].
        :param init: Initial hidden states.
        :return: A list in length [batch_size] which contains tensors in shape [num_frames, output_size].
        """
        length = [_.shape[0] for _ in x]
        x = self.dropout(relu(self.linear1(pad_sequence(x))))
        x = self.rnn(pack_padded_sequence(x, length, enforce_sorted=False), init)[0]
        x = self.linear2(pad_packed_sequence(x)[0])
        return [x[:l, i].clone() for i, l in enumerate(length)]


class RNNWithInit(RNN):
    r"""
    RNN with the initial hidden states regressed from the first output.
    """
    def __init__(self, input_size: int, output_size: int, hidden_size: int, num_rnn_layer: int,
                 rnn_type='lstm', bidirectional=False, dropout=0., load_weight_file: str = None):
        r"""
        Init an RNNWithInit net.

        :param input_size: Input size.
        :param output_size: Output size.
        :param hidden_size: Hidden size for RNN.
        :param num_rnn_layer: Number of RNN layers.
        :param rnn_type: Select from 'rnn', 'lstm', 'gru'.
        :param bidirectional: Whether if the RNN is bidirectional.
        :param dropout: Dropout after the input linear layer and in the rnn.
        :param load_weight_file: If not None and exists, weights will be loaded.
        """
        assert rnn_type == 'lstm' and bidirectional is False
        super().__init__(input_size, output_size, hidden_size, num_rnn_layer, rnn_type, bidirectional, dropout)

        self.init_net = torch.nn.Sequential(
            torch.nn.Linear(output_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size * num_rnn_layer),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size * num_rnn_layer, 2 * (2 if bidirectional else 1) * num_rnn_layer * hidden_size)
        )

        if load_weight_file and os.path.exists(load_weight_file):
            self.load_state_dict(torch.load(load_weight_file))
            self.eval()

    def forward(self, x, _=None):
        r"""
        Forward.

        :param x: A list in length [batch_size] which contains 2-tuple
                  (Tensor[num_frames, input_size], Tensor[output_size]).
        :param _: Not used.
        :return: A list in length [batch_size] which contains tensors in shape [num_frames, output_size].
        """
        x, x_init = list(zip(*x))
        nd, nh = self.rnn.num_layers * (2 if self.rnn.bidirectional else 1), self.rnn.hidden_size
        h, c = self.init_net(torch.stack(x_init)).view(-1, 2, nd, nh).permute(1, 2, 0, 3)
        return super(RNNWithInit, self).forward(x, (h, c))
