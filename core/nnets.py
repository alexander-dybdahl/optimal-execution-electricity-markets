import torch
import torch.nn as nn
import torch.nn.functional as F


class Sine(nn.Module):
    """This class defines the sine activation function as a nn.Module"""
    def __init__(self):
        super(Sine, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class FCnet(nn.Module):
    def __init__(self, layers, activation, use_sync_bn=False, use_batchnorm=True):
        super(FCnet, self).__init__()
        self.T = None  # Set externally
        self.use_sync_bn = use_sync_bn
        self.use_batchnorm = use_batchnorm
        self.state_dim = layers[0] - 1
        if use_batchnorm:
            bn_cls = nn.SyncBatchNorm if use_sync_bn else nn.BatchNorm1d
            self.bn_state = bn_cls(self.state_dim)
        else:
            self.bn_state = None
        self.layers = []
        for i in range(len(layers) - 2):
            self.layers.append(nn.Linear(in_features=layers[i], out_features=layers[i + 1]))
            self.layers.append(activation)
        self.layers.append(nn.Linear(in_features=layers[-2], out_features=layers[-1]))
        self.net = nn.Sequential(*self.layers)

    def forward(self, t, y):
        t_norm = t / self.T
        y_norm = self.bn_state(y) if self.use_batchnorm else y
        return self.net(torch.cat([t_norm, y_norm], dim=1))


class Resnet(nn.Module):
    def __init__(self, layers, activation, stable=False, use_sync_bn=False, use_batchnorm=True):
        super(Resnet, self).__init__()
        self.activation = activation
        self.stable = stable
        self.epsilon = 0.01
        self.input_dim = layers[0]
        self.hidden_dims = layers[1:-1]
        self.output_dim = layers[-1]
        self.use_sync_bn = use_sync_bn
        self.use_batchnorm = use_batchnorm
        self.state_dim = self.input_dim - 1
        if use_batchnorm:
            bn_cls = nn.SyncBatchNorm if use_sync_bn else nn.BatchNorm1d
            self.bn_state = bn_cls(self.state_dim)
        else:
            self.bn_state = None

        self.hidden_layers = nn.ModuleList()
        self.shortcut_layers = nn.ModuleList()

        for i in range(len(self.hidden_dims)):
            in_dim = self.input_dim if i == 0 else self.hidden_dims[i - 1]
            out_dim = self.hidden_dims[i]

            self.hidden_layers.append(nn.Linear(in_dim, out_dim))
            if i > 0:
                self.shortcut_layers.append(nn.Linear(self.input_dim, out_dim))

        self.output_layer = nn.Linear(self.hidden_dims[-1], self.output_dim)

    def forward(self, t, y):
        t_norm = t / self.T
        y_norm = self.bn_state(y) if self.use_batchnorm else y
        u = torch.cat([t_norm, y_norm], dim=1)
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

    def stable_forward(self, layer, out):
        W = layer.weight
        delta = 1 - 2 * self.epsilon
        RtR = torch.matmul(W.t(), W)
        norm = torch.norm(RtR)
        if norm > delta:
            RtR = delta ** 0.5 * RtR / norm**0.5
        A = RtR + torch.eye(RtR.shape[0], device=RtR.device) * self.epsilon
        return F.linear(out, -A, layer.bias)


class FeedForwardSubNet(nn.Module):
    """A feed-forward neural network with granular batch normalization control.
    
    This is the core building block for all subnet-based architectures.
    Supports configurable activation functions and independent BatchNorm control
    for input, hidden layers, and output layers.
    """

    def __init__(self, input_dim, hidden_dims, output_dim=1, activation='ReLU', 
                 use_bn_input=True, use_bn_hidden=True, use_bn_output=True,
                 use_sync_bn=False, device=None, dtype=None):
        super(FeedForwardSubNet, self).__init__()
        self.use_bn_input = use_bn_input
        self.use_bn_hidden = use_bn_hidden
        self.use_bn_output = use_bn_output
        self.use_sync_bn = use_sync_bn

        # Define activation function
        activations = {
            'ReLU': nn.ReLU(),
            'LeakyReLU': nn.LeakyReLU(negative_slope=0.01),
            'Sigmoid': nn.Sigmoid(),
            'Tanh': nn.Tanh(),
            'ELU': nn.ELU(alpha=1.0),
            'GELU': nn.GELU(),
            'SELU': nn.SELU(),
            'SiLU': nn.SiLU(),
            'Softplus': nn.Softplus(),
            'Softsign': nn.Softsign(),
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation function: {activation}")
        self.activation = activations[activation]

        # Create batch normalization layers
        self.bn_layers = nn.ModuleList()
        
        # Input BN
        if use_bn_input:
            bn_cls = nn.SyncBatchNorm if use_sync_bn else nn.BatchNorm1d
            self.bn_layers.append(bn_cls(input_dim, eps=1e-6, momentum=0.01))
        else:
            self.bn_layers.append(None)

        # BN for each hidden layer
        for h in hidden_dims:
            if use_bn_hidden:
                bn_cls = nn.SyncBatchNorm if use_sync_bn else nn.BatchNorm1d
                self.bn_layers.append(bn_cls(h, eps=1e-6, momentum=0.01))
            else:
                self.bn_layers.append(None)

        # Output BN
        if use_bn_output:
            bn_cls = nn.SyncBatchNorm if use_sync_bn else nn.BatchNorm1d
            self.bn_layers.append(bn_cls(output_dim, eps=1e-6, momentum=0.01))
        else:
            self.bn_layers.append(None)

        # Create linear layers
        self.dense_layers = nn.ModuleList()
        in_features = input_dim
        for h in hidden_dims:
            self.dense_layers.append(nn.Linear(in_features, h, bias=False))
            in_features = h
        self.dense_layers.append(nn.Linear(in_features, output_dim, bias=False))

        # Initialize BN parameters
        for bn_layer in self.bn_layers:
            if bn_layer is not None:
                if bn_layer.weight is not None:
                    nn.init.uniform_(bn_layer.weight, 0.1, 0.5)  # gamma
                if bn_layer.bias is not None:
                    nn.init.normal_(bn_layer.bias, 0.0, 0.1)     # beta

    def forward(self, x):
        # Input BN
        if self.bn_layers[0] is not None:
            x = self.bn_layers[0](x)

        # Hidden layers: Linear -> BN -> Activation
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            if self.bn_layers[i + 1] is not None:
                x = self.bn_layers[i + 1](x)
            x = self.activation(x)

        # Final layer -> Output BN
        x = self.dense_layers[-1](x)
        if self.bn_layers[-1] is not None:
            x = self.bn_layers[-1](x)

        return x


class LSTMWithSubnets(nn.Module):
    """LSTM + subnetworks per time step architecture.
    
    Architecture:
    1. Main LSTM processes (time, state) input
    2. LSTM output goes to time-specific subnet
    3. Each time step has its own subnet parameters
    
    Use case: When you want LSTM to learn temporal patterns + specialized processing per time step
    """

    def __init__(self, input_dim, lstm_hidden_size, output_dim, activation, 
                 use_bn_input, use_bn_hidden, use_bn_output, use_sync_bn, 
                 subnet_hidden_dims, num_time_steps, device=None, dtype=None):
        super(LSTMWithSubnets, self).__init__()
        
        self.T = None  # Set externally
        self.state_dim = input_dim - 1  # input_dim includes time dimension
        self.num_time_steps = num_time_steps
        
        # Input batch normalization for state variables
        if use_bn_input:
            bn_cls = nn.SyncBatchNorm if use_sync_bn else nn.BatchNorm1d
            self.bn_state = bn_cls(self.state_dim, affine=False)
        else:
            self.bn_state = None

        # Main LSTM layer
        self.lstm = nn.LSTM(input_dim, lstm_hidden_size, batch_first=True)
        
        # Create subnets for each time step
        self.subnets = nn.ModuleList([
            FeedForwardSubNet(
                input_dim=lstm_hidden_size,
                hidden_dims=subnet_hidden_dims,
                output_dim=output_dim,
                activation=activation,
                use_bn_input=use_bn_input,
                use_bn_hidden=use_bn_hidden,
                use_bn_output=use_bn_output,
                use_sync_bn=use_sync_bn,
                device=device,
                dtype=dtype
            ) for _ in range(self.num_time_steps)
        ])

    def forward(self, t, y):
        # Normalize inputs
        t_norm = t / self.T if self.T is not None else t
        y_norm = self.bn_state(y) if self.bn_state is not None else y
        input_seq = torch.cat([t_norm, y_norm], dim=1)
        
        # LSTM forward pass
        lstm_input = input_seq.unsqueeze(1)  # [batch, seq_len=1, features]
        lstm_out, _ = self.lstm(lstm_input)
        lstm_features = lstm_out.squeeze(1)  # [batch, lstm_hidden_size]
        
        # Determine which time step subnet to use (assuming uniform time steps)
        time_step_idx = torch.clamp(
            (t_norm[0, 0] * self.num_time_steps).long(),
            0, self.num_time_steps - 1
        ).item()
        
        # Process all samples with the same subnet (uniform time step assumption)
        output = self.subnets[time_step_idx](lstm_features)
        
        return output


class SeparateSubnetsPerTime(nn.Module):
    """Simple separate subnetworks per time step (no LSTM).
    
    Architecture:
    1. Input normalization (optional)
    2. Time-specific subnet selection
    3. Each time step has its own subnet parameters
    
    Use case: When you want specialized processing per time step without LSTM overhead
    """

    def __init__(self, input_dim, output_dim, activation, use_bn_input, 
                 use_bn_hidden, use_bn_output, use_sync_bn, subnet_hidden_dims, 
                 num_time_steps, device=None, dtype=None):
        super(SeparateSubnetsPerTime, self).__init__()
        
        self.T = None  # Set externally
        self.state_dim = input_dim - 1  # input_dim includes time dimension
        self.num_time_steps = num_time_steps
        
        # Input batch normalization for state variables
        if use_bn_input:
            bn_cls = nn.SyncBatchNorm if use_sync_bn else nn.BatchNorm1d
            self.bn_state = bn_cls(self.state_dim, affine=False)
        else:
            self.bn_state = None
        
        # Create separate subnet for each time step
        self.subnets = nn.ModuleList([
            FeedForwardSubNet(
                input_dim=input_dim,  # time + state
                hidden_dims=subnet_hidden_dims,
                output_dim=output_dim,
                activation=activation,
                use_bn_input=use_bn_input,
                use_bn_hidden=use_bn_hidden,
                use_bn_output=use_bn_output,
                use_sync_bn=use_sync_bn,
                device=device,
                dtype=dtype
            ) for _ in range(self.num_time_steps)
        ])

    def forward(self, t, y):
        # Normalize inputs
        t_norm = t / self.T if self.T is not None else t
        y_norm = self.bn_state(y) if self.bn_state is not None else y
        input_features = torch.cat([t_norm, y_norm], dim=1)
        
        # Determine which time step subnet to use (assuming uniform time steps)
        time_step_idx = torch.clamp(
            (t_norm[0, 0] * self.num_time_steps).long(),
            0, self.num_time_steps - 1
        ).item()
        
        # Process all samples with the same subnet (uniform time step assumption)
        output = self.subnets[time_step_idx](input_features)
        
        return output
