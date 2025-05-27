import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class LSTMNet(nn.Module):
    def __init__(self, layers, activation):
        super().__init__()
        input_size = layers[0]  # dim + 1 (time + state)
        hidden_sizes = layers[1:-1]
        output_size = layers[-1]

        self.lstm_layers = nn.ModuleList([
            LSTMCell(input_size if i == 0 else hidden_sizes[i - 1], hidden_sizes[i])
            for i in range(len(hidden_sizes))
        ])
        self.out_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.activation = activation

    def forward(self, t, y):
        input_seq = torch.cat([t, y], dim=1).unsqueeze(1)  # shape: [batch, seq_len=1, input_size]
        batch_size = input_seq.size(0)

        h = [torch.zeros(batch_size, layer.hidden_size, device=input_seq.device, requires_grad=True)
            for layer in self.lstm_layers]
        c = [torch.zeros_like(h_i) for h_i in h]

        x = input_seq.squeeze(1)
        for i, lstm_cell in enumerate(self.lstm_layers):
            h[i], (h[i], c[i]) = lstm_cell(x, (h[i], c[i]))
            x = self.activation(h[i])

        return self.out_layer(x)

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        gates = self.i2h(x) + self.h2h(h_prev)
        i, f, g, o = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, (h, c)

class Sine(nn.Module):
    """This class defines the sine activation function as a nn.Module"""
    def __init__(self):
        super(Sine, self).__init__()

    def forward(self, x):
        return torch.sin(x)

class FCnet(nn.Module):

    def __init__(self, layers, activation):
        super(FCnet, self).__init__()

        self.layers = []
        for i in range(len(layers) - 2):
            self.layers.append(nn.Linear(in_features=layers[i], out_features=layers[i + 1]))
            self.layers.append(activation)
        self.layers.append(nn.Linear(in_features=layers[-2], out_features=layers[-1]))

        self.net = nn.Sequential(*self.layers)

    def forward(self, t, y):
        return self.net(torch.cat([t, y], dim=1))

class Resnet(nn.Module):
    def __init__(self, layers, activation, stable=False):
        super(Resnet, self).__init__()
        self.activation = activation
        self.stable = stable
        self.epsilon = 0.01

        self.input_dim = layers[0]
        self.hidden_dims = layers[1:-1]
        self.output_dim = layers[-1]

        self.hidden_layers = nn.ModuleList()
        self.shortcut_layers = nn.ModuleList()

        for i in range(len(self.hidden_dims)):
            in_dim = self.input_dim if i == 0 else self.hidden_dims[i - 1]
            out_dim = self.hidden_dims[i]

            self.hidden_layers.append(nn.Linear(in_dim, out_dim))
            if i > 0:
                self.shortcut_layers.append(nn.Linear(self.input_dim, out_dim))

        self.output_layer = nn.Linear(self.hidden_dims[-1], self.output_dim)

    def stable_forward(self, layer, out):
        W = layer.weight
        delta = 1 - 2 * self.epsilon
        RtR = torch.matmul(W.t(), W)
        norm = torch.norm(RtR)
        if norm > delta:
            RtR = delta ** 0.5 * RtR / norm**0.5
        A = RtR + torch.eye(RtR.shape[0], device=RtR.device) * self.epsilon
        return F.linear(out, -A, layer.bias)

    def forward(self, t, y):
        u = torch.cat([t, y], dim=1)
        out = u

        for i, layer in enumerate(self.hidden_layers):
            shortcut = out

            if self.stable and i > 0:
                out = self.stable_forward(layer, out)
                out = out + self.shortcut_layers[i - 1](u)
            else:
                out = layer(out)

            out = self.activation(out)
            out = out + shortcut if i > 0 else out

        return self.output_layer(out)

