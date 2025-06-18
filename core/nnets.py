import torch
import torch.nn as nn
import torch.nn.functional as F


class FCnet_init(nn.Module):

    def __init__(self, layers, activation, y0=None, rescale_y0=False):
        super(FCnet_init, self).__init__()
        self.rescale_y0 = rescale_y0

        if rescale_y0:
            self.y0=y0
        
        self.layers = []
        for i in range(len(layers) - 2):
            self.layers.append(nn.Linear(in_features=layers[i], out_features=layers[i + 1]))
            self.layers.append(activation)
        self.layers.append(nn.Linear(in_features=layers[-2], out_features=layers[-1]))

        self.net = nn.Sequential(*self.layers)

    def forward(self, y):
        if self.rescale_y0:
            # Only divide by the elements of y0 that are not zero
            y = torch.where(self.y0 != 0, y / self.y0, torch.zeros_like(y))

        return self.net(y)

class LSTMNet(nn.Module):
    def __init__(self, layers, activation, type='LSTM', T=None, input_bn=False, affine=False):
        super().__init__()
        input_size = layers[0]  # dim + 1 (time + state)
        hidden_sizes = layers[1:-1]
        output_size = layers[-1]
        self.T = T
        self.input_bn = input_bn
        
        if input_bn:
            self.input_bn_layer = nn.BatchNorm1d(input_size-1, affine=affine, track_running_stats=True)

        if type == 'ResLSTM':
            self.lstm_layers = nn.ModuleList([
                ResLSTMCell(input_size if i == 0 else hidden_sizes[i - 1], hidden_sizes[i], activation=activation)
                for i in range(len(hidden_sizes))
            ])
        elif type == 'NaisLSTM':
            self.lstm_layers = nn.ModuleList([
                ResLSTMCell(input_size if i == 0 else hidden_sizes[i - 1], hidden_sizes[i], stable=True, activation=activation)
                for i in range(len(hidden_sizes))
            ])
        else:
            self.lstm_layers = nn.ModuleList([
                LSTMCell(input_size if i == 0 else hidden_sizes[i - 1], hidden_sizes[i])
                for i in range(len(hidden_sizes))
            ])
        self.out_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.activation = activation

    def forward(self, t, y):
        if self.input_bn:
            t = t / self.T
            y = self.input_bn_layer(y)

        # Concatenate time and state
        u = torch.cat([t, y], dim=1)  # shape: [batch, input_size]
        batch_size = u.size(0)

        h = [torch.zeros(batch_size, layer.hidden_size, device=u.device, requires_grad=True)
            for layer in self.lstm_layers]
        c = [torch.zeros_like(h_i) for h_i in h]

        x = u.squeeze(1)
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

    def forward(self, u, hidden):
        h_prev, c_prev = hidden
        gates = self.i2h(u) + self.h2h(h_prev)
        i, f, g, o = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, (h, c)

class ResLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, stable=False, activation=nn.Tanh()):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.stable = stable
        self.activation = activation
        self.epsilon = 0.01

        self.i2h = nn.Linear(input_size, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size)
        self.residual = nn.Linear(input_size, hidden_size)

    def stable_forward(self, layer, u):
        # Stabilize the weight matrix W using its spectral norm
        W = layer.weight  # shape: [hidden_size, input_size]
        b = layer.bias

        # Compute spectral norm regularization
        with torch.no_grad():
            W_norm = torch.norm(W, p=2)
            if W_norm > (1 - self.epsilon):
                W = W * ((1 - self.epsilon) / W_norm)

        # Use the stabilized weight for a manual forward pass
        return F.linear(u, W, b)

    def forward(self, u, hidden):
        h_prev, c_prev = hidden
        gates = self.i2h(u) + self.h2h(h_prev)
        i, f, g, o = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c = f * c_prev + i * g
        h_core = o * torch.tanh(c)

        # Inject residual connection
        res = self.residual(u)
        if self.stable:
            res = self.stable_forward(self.residual, u)

        h = h_core + res  # or h = h_core + shortcut(x)
        h = self.activation(h)

        return h, (h, c)

class Sine(nn.Module):
    """This class defines the sine activation function as a nn.Module"""
    def __init__(self):
        super(Sine, self).__init__()

    def forward(self, t, y):
        return self.net(torch.cat([t, y], dim=1))

class FCnet(nn.Module):

    def __init__(self, layers, activation, T=None, input_bn=False, affine=False):
        super(FCnet, self).__init__()
        self.input_bn = input_bn
        
        if input_bn:
            self.T = T
            self.input_bn_layer = nn.BatchNorm1d(layers[0]-1, affine=affine, track_running_stats=True)

        self.layers = []
        for i in range(len(layers) - 2):
            self.layers.append(nn.Linear(in_features=layers[i], out_features=layers[i + 1]))
            self.layers.append(activation)
        self.layers.append(nn.Linear(in_features=layers[-2], out_features=layers[-1]))

        self.net = nn.Sequential(*self.layers)

    def forward(self, t, y):
        if self.input_bn:
            t = t / self.T
            y = self.input_bn_layer(y)

        return self.net(torch.cat([t, y], dim=1))

class Resnet(nn.Module):
    def __init__(self, layers, activation, stable=False, T=None, input_bn=False, affine=False):
        super(Resnet, self).__init__()
        self.activation = activation
        self.stable = stable
        self.input_bn = input_bn

        if input_bn:
            self.T = T
            self.input_bn_layer = nn.BatchNorm1d(layers[0]-1, affine=affine, track_running_stats=True)

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

    def stable_forward(self, layer, u):
        W = layer.weight
        delta = 1 - 2 * self.epsilon
        RtR = torch.matmul(W.t(), W)
        norm = torch.norm(RtR)
        if norm > delta:
            RtR = delta ** 0.5 * RtR / norm**0.5
        A = RtR + torch.eye(RtR.shape[0], device=RtR.device) * self.epsilon
        return F.linear(u, -A, layer.bias)

    def forward(self, t, y):
        if self.input_bn:
            t = t / self.T
            y = self.input_bn_layer(y)

        # Concatenate time and state
        u = torch.cat([t, y], dim=1)  # shape: [batch, input_size]
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

class SeparateSubnets(nn.Module):
    """Architecture with separate subnetworks for each time step.
    
    This architecture creates a different subnetwork (FCnet or Resnet) for each time step,
    similar to the implementation in subnet.py and model.py examples.
    """
    def __init__(self, layers, activation, num_time_steps, T, subnet_type="FC", input_bn=True, affine=False):
        super(SeparateSubnets, self).__init__()
        self.activation = activation
        self.num_time_steps = num_time_steps
        self.T = T
        self.subnet_type = subnet_type
        self.input_bn = input_bn

        if input_bn:
            self.input_bn_layer = nn.BatchNorm1d(layers[0]-1, affine=affine, track_running_stats=True)
        
        input_dim = layers[0]  # dim + 1 (time + state)
        self.subnet_layers = layers[1:]  # layers for each subnet
        # Create a separate subnet for each time step
        self.subnets = nn.ModuleList()
        for _ in range(num_time_steps):
            if subnet_type == "FC":
                subnet = FCnet(
                    layers=[input_dim] + self.subnet_layers,
                    activation=activation
                )
            elif subnet_type == "Resnet":
                subnet = Resnet(
                    layers=[input_dim] + self.subnet_layers,
                    activation=activation,
                    stable=False
                )
            elif subnet_type == "NAISnet":
                subnet = Resnet(
                    layers=[input_dim] + self.subnet_layers,
                    activation=activation,
                    stable=True
                )
            else:
                raise ValueError(f"Unknown subnet type: {subnet_type}")
            self.subnets.append(subnet)
    
    def forward(self, t, y):
        # Determine which subnet to use based on time
        relative_t = t / self.T  # Normalize time to [0, 1]
        time_idx = min(int(relative_t[0, 0] * self.num_time_steps), self.num_time_steps-1)
        
        if self.input_bn:
            t = relative_t
            y = self.input_bn_layer(y)

        # For subnets, we need to create dummy time input since they expect (t, y) format
        dummy_t = torch.zeros_like(t)
        
        # Use the appropriate subnet for this time step
        return self.subnets[time_idx](dummy_t, y)

class LSTMWithSubnets(nn.Module):
    """Architecture with shared LSTM layer followed by separate subnetworks per time step.
    
    This architecture first processes input through a shared LSTM layer, then uses
    different subnetworks (FCnet or Resnet) for each time step.
    """
    def __init__(self, layers, activation, num_time_steps, T, lstm_layers=[64], subnet_type="FC", 
                 lstm_type="LSTM", input_bn=True, affine=False):
        super(LSTMWithSubnets, self).__init__()
        self.activation = activation
        self.num_time_steps = num_time_steps
        self.T = T
        self.lstm_type = lstm_type
        self.subnet_type = subnet_type
        self.input_bn = input_bn

        if input_bn:
            self.input_bn_layer = nn.BatchNorm1d(layers[0]-1, affine=affine, track_running_stats=True)
        
        input_size = layers[0]  # dim + 1 (time + state)
        lstm_hidden_sizes = lstm_layers  # LSTM layer configuration
        self.subnet_layers = layers[1:]  # layers for each subnet
        
        # Create shared LSTM layer(s)
        if lstm_type == 'ResLSTM':
            self.lstm_layers = nn.ModuleList([
                ResLSTMCell(input_size if i == 0 else lstm_hidden_sizes[i - 1], 
                           lstm_hidden_sizes[i], activation=activation)
                for i in range(len(lstm_hidden_sizes))
            ])
        elif lstm_type == 'NaisLSTM':
            self.lstm_layers = nn.ModuleList([
                ResLSTMCell(input_size if i == 0 else lstm_hidden_sizes[i - 1], 
                           lstm_hidden_sizes[i], stable=True, activation=activation)
                for i in range(len(lstm_hidden_sizes))
            ])
        else:  # Standard LSTM
            self.lstm_layers = nn.ModuleList([
                LSTMCell(input_size if i == 0 else lstm_hidden_sizes[i - 1], 
                        lstm_hidden_sizes[i])
                for i in range(len(lstm_hidden_sizes))
            ])
        
        # Create separate subnets for each time step
        # Input to subnets is the output of the last LSTM layer + dummy time dimension (1)
        lstm_output_size = lstm_hidden_sizes[-1]
        # Add 1 to account for dummy time dimension that will be concatenated
        subnet_input_size = lstm_output_size + 1
        self.subnets = nn.ModuleList()
        for i in range(num_time_steps):
            if subnet_type == "FC":
                subnet = FCnet(
                    layers=[subnet_input_size] + self.subnet_layers,
                    activation=activation
                )
            elif subnet_type == "Resnet":
                subnet = Resnet(
                    layers=[subnet_input_size] + self.subnet_layers,
                    activation=activation,
                    stable=False
                )
            elif subnet_type == "NAISnet":
                subnet = Resnet(
                    layers=[subnet_input_size] + self.subnet_layers,
                    activation=activation,
                    stable=True
                )
            else:
                raise ValueError(f"Unknown subnet type: {subnet_type}")
            self.subnets.append(subnet)
        
        # Initialize hidden states (will be reset for each forward pass)
        self.hidden_states = None
        self.cell_states = None
    
    def forward(self, t, y):
        # Determine which subnet to use based on time
        relative_t = t / self.T  # Normalize time to [0, 1]
        time_idx = min(int(relative_t[0, 0] * self.num_time_steps), self.num_time_steps-1)
        
        if self.input_bn:
            t = relative_t
            y = self.input_bn_layer(y)

        # Concatenate time and state
        u = torch.cat([t, y], dim=1)  # shape: [batch, input_size]
        batch_size = u.size(0)
        # Initialize hidden and cell states if needed
        if (self.hidden_states is None or 
            self.hidden_states[0].shape[0] != batch_size or
            self.hidden_states[0].device != u.device):
            
            self.hidden_states = [
                torch.zeros(batch_size, layer.hidden_size, 
                           device=u.device, dtype=u.dtype, requires_grad=True)
                for layer in self.lstm_layers
            ]
            self.cell_states = [torch.zeros_like(h) for h in self.hidden_states]
        
        # Process through LSTM layers
        x = u
        for i, lstm_cell in enumerate(self.lstm_layers):
            if hasattr(lstm_cell, 'hidden_size'):  # Standard LSTM or ResLSTM
                h_new, (h_state, c_state) = lstm_cell(x, (self.hidden_states[i], self.cell_states[i]))
                # TODO: Check if we need to detach states
                self.hidden_states[i] = h_state.detach()
                self.cell_states[i] = c_state.detach()
                x = self.activation(h_new)
            else:
                # For other LSTM types that might have different interfaces
                h_new = lstm_cell(x)
                x = self.activation(h_new)
        
        # For subnets, we need to create dummy time input since they expect (t, y) format
        dummy_t = torch.zeros_like(t)
        
        # Use the appropriate subnet for this time step
        return self.subnets[time_idx](dummy_t, x)
    
    def reset_states(self):
        """Reset LSTM hidden and cell states. Call this between different sequences."""
        self.hidden_states = None
        self.cell_states = None