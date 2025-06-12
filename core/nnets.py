import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMNet(nn.Module):
    def __init__(self, layers, activation, type='LSTM', use_sync_bn=False, use_batchnorm=True):
        super().__init__()
        input_size = layers[0]  # dim + 1 (time + state)
        hidden_sizes = layers[1:-1]
        output_size = layers[-1]
        self.activation = activation
        self.T = None  # Set externally
        self.use_sync_bn = use_sync_bn
        self.use_batchnorm = use_batchnorm
        self.state_dim = input_size - 1
        if use_batchnorm:
            bn_cls = nn.SyncBatchNorm if use_sync_bn else nn.BatchNorm1d
            self.bn_state = bn_cls(self.state_dim, affine=False)
        else:
            self.bn_state = None
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

    def forward(self, t, y):
        t_norm = t / self.T
        y_norm = self.bn_state(y) if self.use_batchnorm else y
        input_seq = torch.cat([t_norm, y_norm], dim=1).unsqueeze(1)  # shape: [batch, seq_len=1, input_size]
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

    def stable_forward(self, layer, x):
        # Stabilize the weight matrix W using its spectral norm
        W = layer.weight  # shape: [hidden_size, input_size]
        b = layer.bias

        # Compute spectral norm regularization
        with torch.no_grad():
            W_norm = torch.norm(W, p=2)
            if W_norm > (1 - self.epsilon):
                W = W * ((1 - self.epsilon) / W_norm)

        # Use the stabilized weight for a manual forward pass
        return F.linear(x, W, b)

    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        gates = self.i2h(x) + self.h2h(h_prev)
        i, f, g, o = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c = f * c_prev + i * g
        h_core = o * torch.tanh(c)

        # Inject residual connection
        res = self.residual(x)
        if self.stable:
            res = self.stable_forward(self.residual, x)

        h = h_core + res  # or h = h_core + shortcut(x)
        h = self.activation(h)

        return h, (h, c)

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
    """A feed-forward neural network with batch normalization layers.
    
    Based on the subnet.py implementation but adapted for the FBSNN framework.
    Supports configurable activation functions, granular batch normalization control, and initialization.
    """

    def __init__(self, input_dim, hidden_dims, output_dim=1, activation='ReLU', 
                 use_bn_input=True, use_bn_hidden=True, use_bn_output=True,
                 use_sync_bn=False, device=None, dtype=None):
        super(FeedForwardSubNet, self).__init__()
        self.use_bn_input = use_bn_input
        self.use_bn_hidden = use_bn_hidden
        self.use_bn_output = use_bn_output
        self.use_sync_bn = use_sync_bn
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype if dtype else torch.float32

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
    """LSTM network with subnetworks per time step as described in the paper.
    
    This implements the architecture from https://ar5iv.labs.arxiv.org/html/1902.03986
    where the main LSTM processes the sequence and each time step has its own subnet.
    """

    def __init__(self, layers, activation='ReLU', use_bn_input=True, use_bn_hidden=True, 
                 use_bn_output=True, use_sync_bn=False, subnet_hidden_dims=None, 
                 num_time_steps=None, device=None, dtype=None):
        super(LSTMWithSubnets, self).__init__()
        
        input_size = layers[0]  # dim + 1 (time + state)
        lstm_hidden_size = layers[1] if len(layers) > 1 else 64
        output_size = layers[-1]
        
        self.T = None  # Set externally
        self.use_bn_input = use_bn_input
        self.use_bn_hidden = use_bn_hidden
        self.use_bn_output = use_bn_output
        self.use_sync_bn = use_sync_bn
        self.state_dim = input_size - 1
        self.num_time_steps = num_time_steps if num_time_steps else 20
        
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype if dtype else torch.float32
        
        # Input batch normalization
        if use_bn_input:
            bn_cls = nn.SyncBatchNorm if use_sync_bn else nn.BatchNorm1d
            self.bn_state = bn_cls(self.state_dim, affine=False)
        else:
            self.bn_state = None

        # Main LSTM layer
        self.lstm = nn.LSTM(input_size, lstm_hidden_size, batch_first=True)
        
        # Subnet configuration
        if subnet_hidden_dims is None:
            subnet_hidden_dims = [lstm_hidden_size // 2, lstm_hidden_size // 2]
        
        # Create subnets for each time step
        self.subnets = nn.ModuleList([
            FeedForwardSubNet(
                input_dim=lstm_hidden_size,
                hidden_dims=subnet_hidden_dims,
                output_dim=output_size,
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
        t_norm = t / self.T if self.T is not None else t
        y_norm = self.bn_state(y) if self.use_bn_input and self.bn_state is not None else y
        input_seq = torch.cat([t_norm, y_norm], dim=1)
        
        batch_size = input_seq.size(0)
        
        # Prepare sequence for LSTM (assuming single time step for now)
        # In practice, you might want to handle sequences differently
        lstm_input = input_seq.unsqueeze(1)  # [batch, seq_len=1, features]
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)
        lstm_features = lstm_out.squeeze(1)  # [batch, lstm_hidden_size]
        
        # Determine which time step to use (based on time input)
        # This is a simplified approach - you might want to implement this differently
        time_step_idx = min(int(t_norm[0, 0].item() * self.num_time_steps), self.num_time_steps - 1)
        
        # Use the appropriate subnet for this time step
        output = self.subnets[time_step_idx](lstm_features)
        
        return output


class NonsharedLSTMModel(nn.Module):
    """LSTM model with separate subnets for each time step (non-shared parameters).
    
    This is the main architecture described in the paper where each time step
    has its own subnet for better approximation capability.
    """

    def __init__(self, config, dynamics, activation='ReLU', use_bn_input=True, 
                 use_bn_hidden=True, use_bn_output=True, use_sync_bn=False):
        super(NonsharedLSTMModel, self).__init__()
        
        self.dynamics = dynamics
        self.dim = dynamics.dim
        self.num_time_steps = dynamics.N
        self.activation = activation
        self.use_bn_input = use_bn_input
        self.use_bn_hidden = use_bn_hidden
        self.use_bn_output = use_bn_output
        self.use_sync_bn = use_sync_bn
        
        # Get configuration from config dict or use defaults
        lstm_hidden_size = config.get('lstm_hidden_size', 64)
        subnet_hidden_dims = config.get('subnet_hidden_dims', [32, 32])
        
        # Main LSTM for processing time series
        input_size = self.dim + 1  # state + time
        self.lstm = nn.LSTM(input_size, lstm_hidden_size, batch_first=True)
        
        # Initial value network (Y_0)
        self.y_init_net = FeedForwardSubNet(
            input_dim=self.dim,  # only state, no time
            hidden_dims=subnet_hidden_dims,
            output_dim=1,
            activation=activation,
            use_bn_input=use_bn_input,
            use_bn_hidden=use_bn_hidden,
            use_bn_output=use_bn_output,
            use_sync_bn=use_sync_bn
        )
        
        # Subnet for each time step (Z networks)
        self.z_subnets = nn.ModuleList([
            FeedForwardSubNet(
                input_dim=lstm_hidden_size + self.dim + 1,  # LSTM output + state + time
                hidden_dims=subnet_hidden_dims,
                output_dim=self.dim,  # gradient dimension
                activation=activation,
                use_bn_input=use_bn_input,
                use_bn_hidden=use_bn_hidden,
                use_bn_output=use_bn_output,
                use_sync_bn=use_sync_bn
            ) for _ in range(self.num_time_steps)
        ])
        
        self.T = dynamics.T

    def forward(self, t, y):
        """Forward pass for single time step evaluation."""
        # This is for compatibility with existing FBSNN interface
        batch_size = y.size(0)
        
        # Normalize time
        t_norm = t / self.T if self.T is not None else t
        
        # Create input sequence
        input_seq = torch.cat([t_norm, y], dim=1).unsqueeze(1)  # [batch, 1, dim+1]
        
        # LSTM forward
        lstm_out, _ = self.lstm(input_seq)
        lstm_features = lstm_out.squeeze(1)  # [batch, lstm_hidden_size]
        
        # Determine time step index
        time_step_idx = min(int(t_norm[0, 0].item() * self.num_time_steps), self.num_time_steps - 1)
        
        # Combine LSTM features with input for subnet
        subnet_input = torch.cat([lstm_features, t_norm, y], dim=1)
        
        # Use appropriate subnet
        output = self.z_subnets[time_step_idx](subnet_input)
        
        return output
    
    def forward_sequence(self, t_paths, x_paths):
        """Forward pass for full sequence (for training)."""
        batch_size, num_steps, _ = t_paths.shape
        
        # Initialize Y_0
        y_init = self.y_init_net(x_paths[:, 0, :])  # [batch, 1]
        
        # Prepare sequence for LSTM
        t_norm = t_paths / self.T if self.T is not None else t_paths
        lstm_input = torch.cat([t_norm, x_paths], dim=2)  # [batch, num_steps, dim+1]
        
        # LSTM forward pass for entire sequence
        lstm_out, _ = self.lstm(lstm_input)  # [batch, num_steps, lstm_hidden_size]
        
        # Process each time step with its corresponding subnet
        z_outputs = []
        for t in range(num_steps):
            subnet_input = torch.cat([
                lstm_out[:, t, :],  # LSTM output at time t
                t_norm[:, t, :],    # normalized time
                x_paths[:, t, :]    # state at time t
            ], dim=1)
            
            z_t = self.z_subnets[t](subnet_input)
            z_outputs.append(z_t)
        
        z_sequence = torch.stack(z_outputs, dim=1)  # [batch, num_steps, dim]
        
        return y_init, z_sequence

class SeparateSubnetsPerTime(nn.Module):
    """Simple architecture with separate subnetworks per time step (no LSTM).
    
    This provides a clean implementation where each time step has its own 
    feed-forward subnet without any shared LSTM processing.
    """

    def __init__(self, layers, activation='ReLU', use_bn_input=True, use_bn_hidden=True, 
                 use_bn_output=True, use_sync_bn=False, subnet_hidden_dims=None, 
                 num_time_steps=None, device=None, dtype=None):
        super(SeparateSubnetsPerTime, self).__init__()
        
        input_size = layers[0]  # dim + 1 (time + state)
        output_size = layers[-1]
        
        self.T = None  # Set externally
        self.state_dim = input_size - 1
        self.num_time_steps = num_time_steps if num_time_steps else 20
        
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype if dtype else torch.float32
        
        # Input batch normalization for state variables
        if use_bn_input:
            bn_cls = nn.SyncBatchNorm if use_sync_bn else nn.BatchNorm1d
            self.bn_state = bn_cls(self.state_dim, affine=False)
        else:
            self.bn_state = None
        
        # Subnet configuration
        if subnet_hidden_dims is None:
            subnet_hidden_dims = [64, 64]  # Default hidden dimensions
        
        # Create separate subnet for each time step
        self.subnets = nn.ModuleList([
            FeedForwardSubNet(
                input_dim=input_size,  # time + state
                hidden_dims=subnet_hidden_dims,
                output_dim=output_size,
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
        # Normalize time and state
        t_norm = t / self.T if self.T is not None else t
        y_norm = self.bn_state(y) if self.bn_state is not None else y
        input_features = torch.cat([t_norm, y_norm], dim=1)
        
        # Determine which time step to use
        time_step_idx = torch.clamp(
            (t_norm[:, 0] * self.num_time_steps).long(),
            0, self.num_time_steps - 1
        )
        
        # For batch processing, we need to handle different time steps
        # This is a simplified approach - for production, you might want more sophisticated batching
        if len(torch.unique(time_step_idx)) == 1:
            # All samples are at the same time step
            subnet_idx = time_step_idx[0].item()
            output = self.subnets[subnet_idx](input_features)
        else:
            # Samples are at different time steps - process separately
            batch_size = input_features.size(0)
            output_dim = self.subnets[0].dense_layers[-1].out_features
            output = torch.zeros(batch_size, output_dim, device=input_features.device, dtype=input_features.dtype)
            
            for i in range(batch_size):
                subnet_idx = time_step_idx[i].item()
                output[i:i+1] = self.subnets[subnet_idx](input_features[i:i+1])
        
        return output