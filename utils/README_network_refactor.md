# Network Architecture Refactor

This document describes the refactored network architecture for the BraTS training project, which now uses a configuration-driven approach with clean code practices.

## Overview

The network architecture has been refactored to:

- Extract all configuration variables to `config.yaml`
- Apply clean code practices with proper documentation
- Use type hints and validation
- Implement modular design patterns
- Add comprehensive logging

## Files Structure

### Configuration Files

- `config.yaml`: Main configuration file with all network parameters
- `utils/network_config.py`: Network configuration utility and validation

### Network Architecture Files

- `msHead_3D/network_backbone.py`: Main network architecture (MSHEAD_ATTN)
- `msHead_3D/mra_transformer.py`: Multi-Resolution Attention Transformer
- `msHead_3D/mra_helper.py`: Helper modules and building blocks
- `msHead_3D/multi_scale_head.py`: Multi-scale attention mechanisms

### Training Files

- `train.py`: Updated training script using configuration system
- `utils/logger_setup.py`: Logging configuration utility

## Configuration System

### Network Configuration Structure

The network configuration is defined in `config.yaml` under the `network` section:

```yaml
network:
  # Model architecture
  model_type: "MSHEAD_ATTN"

  # Input/Output configuration
  in_channels: 4
  out_channels: 4
  img_size: [128, 128, 128]
  patch_size: 2
  spatial_dims: 3

  # Transformer configuration
  transformer:
    embed_dims: [48, 96, 192, 384]
    depths: [2, 2, 2, 2]
    num_heads: [3, 6, 12, 24]
    mlp_ratios: [4, 4, 4, 4]
    qkv_bias: true
    qk_scale: null
    drop_rate: 0.0
    attn_drop_rate: 0.0
    drop_path_rate: 0.1
    patch_norm: false
    norm_layer: "LayerNorm"
    norm_eps: 1e-6

  # Attention configuration
  attention:
    window_size: 6
    use_relative_position_bias: true
    use_checkpoint: false

  # Channel calibration configuration
  channel_calibration:
    enabled: false
    reduction_ratio: 4
    norm_layer: "InstanceNorm3d"

  # Decoder configuration
  decoder:
    idwt:
      wavelet: "db1"
      kernel_size: 3
      norm_name: "instance"
      res_block: true

    projection_upsample:
      use_double_conv: true
      residual: true
      stride_4: 4
      stride_2: 2

    final_decoder:
      kernel_size: 3
      upsample_kernel_size: 2
      norm_name: "instance"
      res_block: true

  # Initialization configuration
  initialization:
    weight_std: 0.02
    bias_constant: 0.0
    layer_norm_weight: 1.0
    layer_norm_bias: 0.0

  # Wavelet transform configuration
  wavelet:
    wavelet_type: "db1"
    mode: "zero"

  # Multi-scale head configuration
  multi_scale_head:
    window_size: 6
    use_relative_position_bias: true
    softmax_dim: -1
```

### Configuration Validation

The `NetworkConfig` class in `utils/network_config.py` provides:

- Automatic validation of configuration parameters
- Default values for missing parameters
- Type checking and error handling
- Easy access to configuration values

## Architecture Components

### 1. MSHEAD_ATTN (Main Network)

The main network architecture that combines:

- Multi-scale transformer backbone
- Channel calibration (optional)
- IDWT-based decoder
- Projection upsampling
- Feature fusion

**Key Features:**

- Configuration-driven architecture
- Modular design with separate initialization methods
- Type hints and comprehensive documentation
- Optional channel calibration
- Flexible decoder configuration

### 2. MRATransformer (Backbone)

Multi-Resolution Attention Transformer that provides:

- Hierarchical feature extraction
- Multi-scale attention mechanisms
- Configurable transformer stages
- Drop path regularization

**Key Features:**

- Configurable number of stages
- Flexible embedding dimensions
- Configurable attention heads
- Stochastic depth support

### 3. Building Blocks

#### ChannelCalibration

- SENet-style channel attention
- Configurable reduction ratio
- Multiple normalization options
- Residual connections

#### ProjectionUpsample

- Learnable upsampling
- Configurable stride and residual connections
- Double convolution option
- Group normalization

#### IDWT Blocks

- Wavelet-based upsampling
- Configurable wavelet types
- Multi-stage processing
- Residual connections

## Usage Examples

### Basic Model Creation

```python
from config import config
from utils.network_config import get_network_config
from msHead_3D.network_backbone import create_mshead_model

# Get network configuration
network_config = get_network_config(config.__dict__)

# Create model
model = create_mshead_model(network_config.get_model_kwargs())
```

### Custom Configuration

```python
# Custom network configuration
custom_config = {
    'network': {
        'model_type': 'MSHEAD_ATTN',
        'in_channels': 4,
        'out_channels': 4,
        'img_size': [96, 96, 96],
        'patch_size': 2,
        'transformer': {
            'embed_dims': [32, 64, 128, 256],
            'depths': [2, 2, 2, 2],
            'num_heads': [2, 4, 8, 16],
            'drop_path_rate': 0.1
        }
    }
}

network_config = get_network_config(custom_config)
model = create_mshead_model(network_config.get_model_kwargs())
```

### Training Integration

```python
from train import BraTSTrainer

# The trainer automatically uses the configuration
trainer = BraTSTrainer(
    env_type=config.env,
    max_epochs=config.max_epoch,
    batch_size=config.batch_size,
    # ... other parameters
)

# Model is created automatically with configuration
trainer.train(train_dataset=train_ds, val_dataset=val_ds)
```

## Configuration Options

### Model Variants

You can easily create different model variants by modifying the configuration:

#### Lightweight Model

```yaml
network:
  transformer:
    embed_dims: [32, 64, 128, 256]
    depths: [1, 1, 1, 1]
    num_heads: [2, 4, 8, 16]
```

#### Large Model

```yaml
network:
  transformer:
    embed_dims: [64, 128, 256, 512]
    depths: [3, 3, 3, 3]
    num_heads: [4, 8, 16, 32]
```

### Channel Calibration

Enable/disable channel calibration:

```yaml
network:
  channel_calibration:
    enabled: true
    reduction_ratio: 8
    norm_layer: "BatchNorm3d"
```

### Decoder Configuration

Customize decoder behavior:

```yaml
network:
  decoder:
    idwt:
      wavelet: "db2"
      kernel_size: 5
    projection_upsample:
      use_double_conv: false
      residual: false
```

## Best Practices

### 1. Configuration Management

- Always validate configuration before model creation
- Use meaningful default values
- Document configuration parameters
- Version control configuration files

### 2. Model Architecture

- Keep modules small and focused
- Use type hints for better code clarity
- Implement proper error handling
- Add comprehensive logging

### 3. Training Integration

- Save configuration with checkpoints
- Log model architecture details
- Validate configuration at startup
- Use configuration for model loading

### 4. Code Organization

- Separate configuration from implementation
- Use factory patterns for model creation
- Implement proper inheritance hierarchies
- Add comprehensive documentation

## Migration Guide

### From Old Code

If you're migrating from the old hardcoded approach:

1. **Extract Parameters**: Move hardcoded parameters to `config.yaml`
2. **Update Imports**: Use new configuration utilities
3. **Validate Configuration**: Ensure all required parameters are present
4. **Test Thoroughly**: Verify model behavior matches expectations

### Example Migration

**Old Code:**

```python
model = MSHEAD_ATTN(
    img_size=(128, 128, 128),
    patch_size=2,
    in_chans=4,
    out_chans=4,
    depths=[2, 2, 2, 2],
    feat_size=[48, 96, 192, 384],
    num_heads=[3, 6, 12, 24],
    drop_path_rate=0.1,
    use_checkpoint=False,
)
```

**New Code:**

```python
network_config = get_network_config(config.__dict__)
model = create_mshead_model(network_config.get_model_kwargs())
```

## Troubleshooting

### Common Issues

1. **Configuration Validation Errors**

   - Check that all required parameters are present
   - Ensure parameter types match expected values
   - Verify list lengths are consistent

2. **Model Creation Errors**

   - Validate configuration before model creation
   - Check for incompatible parameter combinations
   - Ensure all dependencies are available

3. **Training Issues**
   - Verify configuration is saved with checkpoints
   - Check that model architecture matches expectations
   - Validate input/output dimensions

### Debugging Tips

1. **Enable Debug Logging**

   ```yaml
   logging:
     log_level_console: "debug"
   ```

2. **Validate Configuration**

   ```python
   network_config = get_network_config(config.__dict__)
   print(network_config)
   ```

3. **Check Model Architecture**
   ```python
   model = create_mshead_model(network_config.get_model_kwargs())
   print(model)
   ```

## Future Enhancements

### Planned Features

- Support for more model variants
- Automatic hyperparameter optimization
- Configuration templates for common use cases
- Enhanced validation and error reporting

### Extension Points

- Custom attention mechanisms
- Additional normalization layers
- New upsampling strategies
- Alternative decoder architectures

## Contributing

When contributing to the network architecture:

1. **Follow Configuration Pattern**: Add new parameters to `config.yaml`
2. **Update Validation**: Add validation rules in `NetworkConfig`
3. **Document Changes**: Update this README and code comments
4. **Add Tests**: Include tests for new functionality
5. **Maintain Backward Compatibility**: Ensure existing configurations still work
