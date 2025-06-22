import torch
import torch.nn as nn
import torch.nn.functional as F

class FCnet_init(nn.Module):

    def __init__(self, layers, activation, y0=None, rescale_y0=False, strong_grad_output=False, scale_output=1.0):
        super(FCnet_init, self).__init__()
        self.rescale_y0 = rescale_y0
        self.strong_grad_output = strong_grad_output
        if self.strong_grad_output:
            self.scale_output = nn.Parameter(torch.tensor(scale_output))

        if rescale_y0:
            self.y0=y0
        
        self.layers = []
        for i in range(len(layers) - 2):
            self.layers.append(nn.Linear(in_features=layers[i], out_features=layers[i + 1]))
            self.layers.append(activation)
        self.final_linear = nn.Linear(in_features=layers[-2], out_features=layers[-1])
        
        self.net = nn.Sequential(*self.layers)

    def forward(self, y):
        if self.rescale_y0:
            # Only divide by the elements of y0 that are not zero
            y = torch.where(self.y0 != 0, y / self.y0, torch.zeros_like(y))

        x = self.net(y)
        out = self.final_linear(x)

        if self.strong_grad_output:
            # Apply a scaled identity to avoid gradient vanishing near 0
            return self.scale_output * out
        else:
            return out

class LSTMNet(nn.Module):
    def __init__(self, layers, activation, type='lstm', T=None, input_bn=False, affine=False):
        super().__init__()
        input_size = layers[0]  # dim + 1 (time + state)
        hidden_sizes = layers[1:-1]
        output_size = layers[-1]
        self.T = T
        self.input_bn = input_bn
        
        if input_bn:
            self.input_bn_layer = nn.BatchNorm1d(input_size-1, affine=affine, track_running_stats=True)

        if type == 'reslstm':
            self.lstm_layers = nn.ModuleList([
                ResLSTMCell(input_size if i == 0 else hidden_sizes[i - 1], hidden_sizes[i], activation=activation)
                for i in range(len(hidden_sizes))
            ])
        elif type == 'naislstm':
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
    def __init__(self, layers, activation, num_time_steps, T, subnet_type="fc", input_bn=True, affine=False):
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
            if subnet_type == "fc":
                subnet = FCnet(
                    layers=[input_dim] + self.subnet_layers,
                    activation=activation
                )
            elif subnet_type == "resnet":
                subnet = Resnet(
                    layers=[input_dim] + self.subnet_layers,
                    activation=activation,
                    stable=False
                )
            elif subnet_type == "naisnet":
                subnet = Resnet(
                    layers=[input_dim] + self.subnet_layers,
                    activation=activation,
                    stable=True
                )
            else:
                raise ValueError(f"Unknown subnet type: {subnet_type.upper()}")
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
    def __init__(self, layers, activation, num_time_steps, T, lstm_layers=[64], subnet_type="fc", 
                 lstm_type="lstm", input_bn=True, affine=False):
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
        if lstm_type == 'reslstm':
            self.lstm_layers = nn.ModuleList([
                ResLSTMCell(input_size if i == 0 else lstm_hidden_sizes[i - 1], 
                           lstm_hidden_sizes[i], activation=activation)
                for i in range(len(lstm_hidden_sizes))
            ])
        elif lstm_type == 'naislstm':
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
            if subnet_type == "fc":
                subnet = FCnet(
                    layers=[subnet_input_size] + self.subnet_layers,
                    activation=activation
                )
            elif subnet_type == "resnet":
                subnet = Resnet(
                    layers=[subnet_input_size] + self.subnet_layers,
                    activation=activation,
                    stable=False
                )
            elif subnet_type == "naisnet":
                subnet = Resnet(
                    layers=[subnet_input_size] + self.subnet_layers,
                    activation=activation,
                    stable=True
                )
            else:
                raise ValueError(f"Unknown subnet type: {subnet_type.upper()}")
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

class UncertaintyWeightedLoss(nn.Module):
    def __init__(self, task_weights):
        """
        task_weights: dict mapping task name -> importance weight (e.g., {"Y0_loss": 2.0, "reg_loss": 0.5})
        """
        super().__init__()
        self.task_weights = task_weights

        # Learnable log variances for each task
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(0.0)) for name in task_weights.keys()
        })

    def forward(self, losses_dict):
        total_loss = 0.0
        for name, loss in losses_dict.items():
            log_var = self.log_vars[name]
            weight = self.task_weights.get(name, 1.0)
            precision = torch.exp(-log_var)

            # Weighted version of the uncertainty formulation:
            # weight * (1/σ² * loss + log σ)
            weighted = weight * (precision * loss + log_var)
            total_loss += weighted

        return total_loss


class GradNorm(nn.Module):
    """
    Implementation of GradNorm algorithm (https://arxiv.org/abs/1711.02257) for adaptively 
    balancing multiple losses during training.

    GradNorm dynamically adjusts task weights based on the relative magnitudes of gradients
    from different tasks, helping prevent one task from dominating the learning process.
    """
    def __init__(self, task_weights, alpha=1.5, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            task_weights: Dictionary mapping task name -> initial weight (e.g., {"lambda_Y0": 1.0, "lambda_cost": 0.5})
            alpha: Strength of gradient normalization. Higher values increase the dominance of 
                  tasks with larger relative losses (paper used values from 0.5 to 1.5)
            device: Device to place tensors on (e.g., 'cpu', 'cuda:0')
        """
        super().__init__()
        self.alpha = alpha
        self.num_tasks = len(task_weights)
        self.device = torch.device(device)
        
        # Convert task weights to a tensor for easier manipulation
        loss_keys = list(task_weights.keys())
        loss_weights = torch.tensor([float(task_weights[k]) for k in loss_keys], dtype=torch.float32, device=device)
        
        # Store the order of loss keys for consistent access
        self.loss_names = loss_keys
        
        # Keep only tasks with non-zero weights
        active_indices = []
        active_names = []
        for i, name in enumerate(self.loss_names):
            if task_weights[name] > 0:
                active_indices.append(i)
                active_names.append(name)
                
        self.active_indices = active_indices
        self.active_names = active_names
        self.num_active_tasks = len(active_indices)
        
        # Initialize learnable weights - we only learn weights for active tasks
        self.loss_weights = nn.Parameter(loss_weights[active_indices].clone())
        
        # Save initial weights sum for renormalization
        self.initial_sum = self.loss_weights.sum().item()
        
        # Store for initial losses (will be set on first forward pass)
        self.register_buffer('initial_losses', torch.zeros(len(active_indices), dtype=torch.float32, device=device))
        self.register_buffer('initted', torch.tensor(False, device=device))
        
        # For tracking training progress
        self.iteration = 0

        self.to(torch.device(device))
        
    def forward(self, losses_dict, shared_parameters):
        """
        Calculate the GradNorm loss for weight updates.
        
        Args:
            losses_dict: Dictionary of unweighted task losses
            shared_parameters: Parameters to compute grad norms from (typically backbone network's last layer)
        
        Returns:
            Tuple of (grad_norm_loss, weighted_task_loss)
        """
        # Extract active losses for training
        active_losses = []
        for name in self.active_names:
            active_losses.append(losses_dict[name])
        
        losses = torch.stack(active_losses)
        
        # Update initial losses on first iteration
        if not self.initted:
            self.initial_losses.copy_(losses.detach())
            self.initted.fill_(True)
        
        # Get relative inverse training rates 
        # L(t)/L(0) for each task
        loss_ratios = losses.detach() / (self.initial_losses + 1e-12)
        
        # Compute the average L2 norm of the gradient of the loss 
        # for each task with respect to the shared parameters
        grad_norms = []
        for i, loss in enumerate(losses):
            # Compute weighted loss
            weighted_loss = self.loss_weights[i] * loss
            
            # Get gradients
            grads = torch.autograd.grad(
                weighted_loss,
                shared_parameters,
                retain_graph=True,
                create_graph=True
            )[0]
            
            # Compute L2 norm of gradients
            grad_norm = torch.norm(grads, p=2)
            grad_norms.append(grad_norm)
        
        grad_norms = torch.stack(grad_norms)
        
        # Average L2 norm across all tasks
        mean_norm = grad_norms.mean()
        
        # Calculate relative inverse training rate
        total = loss_ratios.sum() + 1e-12
        inverse_rate = loss_ratios / (total / len(loss_ratios))
        
        # Calculate target for each grad norm
        target_grad_norms = mean_norm * (inverse_rate ** self.alpha)
        
        # GradNorm loss is the L1 distance between the gradient norms and their targets
        grad_norm_loss = torch.abs(grad_norms - target_grad_norms.detach()).sum()
        
        # The weighted task loss for standard backpropagation - keep gradient flow
        # We use the detached weights but keep the gradients flowing through the losses
        weighted_task_loss = (self.loss_weights.detach() * losses).sum()
        
        self.iteration += 1
        
        return grad_norm_loss, weighted_task_loss
    
    def update_weights(self):
        """
        Renormalizes the weights after their gradients have been computed.
        Call this before optimizer.step() to ensure weights maintain proper scaling.
        """
        # Renormalize weights to sum to the original sum
        with torch.no_grad():
            # Make sure we have positive weights after gradient update
            # (negative weights could cause issues with loss balancing)
            # Softplus ensures all weights are positive
            self.loss_weights.data = F.softplus(self.loss_weights.data)
            
            # Renormalize to maintain original sum
            normalize_coeff = self.initial_sum / (self.loss_weights.sum() + 1e-12)
            self.loss_weights.mul_(normalize_coeff)
    
    def get_loss_weights(self):
        """
        Returns the current loss weights dictionary with all tasks.
        Tasks that aren't being actively weighted will have zeros.
        """
        # Create a full weights tensor initialized with zeros
        full_weights = torch.zeros(self.num_tasks, device=self.device)
        
        # Fill in the active weights
        for i, idx in enumerate(self.active_indices):
            full_weights[idx] = self.loss_weights[i].item()
        
        # Convert to dictionary
        weights_dict = {name: full_weights[i].item() for i, name in enumerate(self.loss_names)}
        return weights_dict

