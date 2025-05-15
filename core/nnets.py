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

    def __init__(self, layers, activation, stable):
        super(Resnet, self).__init__()

        self.layer1 = nn.Linear(in_features=layers[0], out_features=layers[1])
        self.layer2 = nn.Linear(in_features=layers[1], out_features=layers[2])
        self.layer2_input = nn.Linear(in_features=layers[0], out_features=layers[2])
        self.layer3 = nn.Linear(in_features=layers[2], out_features=layers[3])
        self.layer3_input = nn.Linear(in_features=layers[0], out_features=layers[3])
        self.layer4 = nn.Linear(in_features=layers[3], out_features=layers[4])
        self.layer4_input = nn.Linear(in_features=layers[0], out_features=layers[4])
        self.layer5 = nn.Linear(in_features=layers[4], out_features=layers[5])

        self.activation = activation

        self.epsilon = 0.01
        self.stable = stable

    def stable_forward(self, layer, out):  # Building block for the NAIS-Net
        weights = layer.weight
        delta = 1 - 2 * self.epsilon
        RtR = torch.matmul(weights.t(), weights)
        norm = torch.norm(RtR)
        if norm > delta:
            RtR = delta ** (1 / 2) * RtR / (norm ** (1 / 2))
        A = RtR + torch.eye(RtR.shape[0], device=RtR.device) * self.epsilon

        return F.linear(out, -A, layer.bias)

    def forward(self, t, y):
        inp = torch.cat([t, y], dim=1)
        
        u = inp

        out = self.layer1(inp)
        out = self.activation(out)

        shortcut = out
        if self.stable:
            out = self.stable_forward(self.layer2, out)
            out = out + self.layer2_input(u)
        else:
            out = self.layer2(out)
        out = self.activation(out)
        out = out + shortcut

        shortcut = out
        if self.stable:
            out = self.stable_forward(self.layer3, out)
            out = out + self.layer3_input(u)
        else:
            out = self.layer3(out)
        out = self.activation(out)
        out = out + shortcut

        shortcut = out
        if self.stable:
            out = self.stable_forward(self.layer4, out)
            out = out + self.layer4_input(u)
        else:
            out = self.layer4(out)

        out = self.activation(out)
        out = out + shortcut

        out = self.layer5(out)

        return out
