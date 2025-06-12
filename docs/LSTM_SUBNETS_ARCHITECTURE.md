# LSTM with Subnetworks Architecture

This document describes the new LSTM-based neural network architectures implemented for the FBSNN solver, based on the paper "Deep BSDE Solvers for High-Dimensional PDEs" (https://ar5iv.labs.arxiv.org/html/1902.03986).

## Overview

The implementation provides two new architectures that combine LSTM networks with feed-forward subnetworks per time step:

1. **LSTMWithSubnets**: A main LSTM with shared subnetworks
2. **NonsharedLSTM**: A complete implementation with separate subnets for each time step (recommended)

## Architecture Details

### NonsharedLSTM (Recommended)

This is the main architecture described in the paper. It consists of:

- **Main LSTM**: Processes the time series of (time, state) inputs
- **Initial Value Network (Y₀)**: Computes the initial value function
- **Time-specific Subnets (Z networks)**: One subnet for each time step to compute gradients

#### Key Features:
- Each time step has its own subnet with independent parameters
- Supports batch normalization within subnets
- Configurable activation functions
- Adaptive to different problem dimensions

### LSTMWithSubnets

A simpler variant where:
- Main LSTM processes inputs
- Shared subnets are used based on time step index
- Suitable for problems with limited computational resources

## Configuration Parameters

### Required Parameters:
- `architecture`: Set to "NonsharedLSTM" or "LSTMWithSubnets"
- `Y_layers`: Network layer dimensions [lstm_hidden_size, output_dim]

### Optional Parameters:
- `subnet_hidden_dims`: Hidden layer dimensions for subnets (default: [32, 32])
- `lstm_hidden_size`: LSTM hidden state size (default: 64)
- `use_batchnorm`: Enable batch normalization in subnets (default: true)
- `activation`: Activation function for subnets (default: "ReLU")

### Supported Activation Functions:
- ReLU, LeakyReLU, Sigmoid, Tanh, ELU, GELU, SELU, SiLU, Softplus, Softsign

## Example Configurations

### NonsharedLSTM Configuration:
```json
{
  "architecture": "NonsharedLSTM",
  "activation": "Tanh",
  "Y_layers": [64, 1],
  "subnet_hidden_dims": [48, 32],
  "lstm_hidden_size": 128,
  "use_batchnorm": true
}
```

### LSTMWithSubnets Configuration:
```json
{
  "architecture": "LSTMWithSubnets", 
  "activation": "ReLU",
  "Y_layers": [64, 1],
  "subnet_hidden_dims": [32, 32],
  "use_batchnorm": true
}
```

## Implementation Details

### Batch Normalization
- Applied within each subnet (input, hidden layers, output)
- Uses momentum=0.01 and eps=1e-6 for stability
- Supports synchronized batch normalization for distributed training
- Gamma initialized uniformly in [0.1, 0.5], beta initialized normally (0.0, 0.1)

### Memory Efficiency
- Subnets are created only when needed
- Supports gradient checkpointing for large networks
- Efficient parameter sharing where possible

### Training Considerations
- Works with both supervised and unsupervised training modes
- Compatible with existing loss functions (Y, dY, dYt, Terminal, PINN losses)
- Supports all existing optimization strategies

## Performance Notes

### Computational Complexity:
- **NonsharedLSTM**: O(N × subnet_params) where N is number of time steps
- **LSTMWithSubnets**: O(subnet_params) - parameter sharing reduces memory

### Memory Usage:
- Higher memory usage due to per-timestep subnets
- Batch normalization adds additional parameters
- Consider reducing subnet_hidden_dims for memory-constrained environments

### Recommended Settings:
- For high-dimensional problems (dim > 10): lstm_hidden_size ≥ 128
- For real-time applications: Use LSTMWithSubnets with smaller subnets
- For accuracy-critical tasks: Use NonsharedLSTM with larger subnets

## Migration from Existing Architectures

To migrate from existing LSTM architectures:

1. Change `architecture` to "NonsharedLSTM"
2. Add `subnet_hidden_dims` and `lstm_hidden_size` to config
3. Adjust `Y_layers` to reflect the simpler output structure
4. Consider enabling batch normalization for better performance

Example migration:
```json
// Before (standard LSTM)
{
  "architecture": "LSTM",
  "Y_layers": [256, 256, 256, 256, 1]
}

// After (NonsharedLSTM)
{
  "architecture": "NonsharedLSTM", 
  "Y_layers": [128, 1],
  "lstm_hidden_size": 128,
  "subnet_hidden_dims": [64, 64]
}
```

## Troubleshooting

### Common Issues:
1. **Memory errors**: Reduce subnet_hidden_dims or lstm_hidden_size
2. **Convergence issues**: Try different activation functions or enable batch normalization
3. **Performance degradation**: Ensure sufficient LSTM hidden size for your problem dimension

### Debug Tips:
- Monitor subnet output distributions during training
- Check batch normalization statistics for unstable training
- Verify time step indexing is correct for your problem setup
