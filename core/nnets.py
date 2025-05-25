import torch
import torch.nn as nn
import torch.nn.functional as F

class YLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(YLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, t_seq, y_seq):
        inp = torch.cat([t_seq, y_seq], dim=2)
        lstm_out, _ = self.lstm(inp)
        return self.output_layer(lstm_out)

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

