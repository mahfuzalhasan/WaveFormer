#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Simple test script to validate configuration structure without requiring PyTorch.
"""

import sys
import os

# Add the parent directory to the path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_config_structure():
    """Test the configuration structure without importing PyTorch-dependent modules."""
    try:
        from config import config
        print("✓ Config loaded successfully")
        
        # Check if network configuration exists
        if hasattr(config, 'network'):
            print("✓ Network configuration found")
            
            network = config.network
            required_keys = ['model_type', 'in_channels', 'out_channels', 'img_size', 'transformer']
            
            for key in required_keys:
                if key in network:
                    print(f"✓ {key}: {network[key]}")
                else:
                    print(f"✗ Missing required key: {key}")
                    return False
            
            # Check transformer configuration
            transformer = network['transformer']
            transformer_keys = ['embed_dims', 'depths', 'num_heads']
            
            for key in transformer_keys:
                if key in transformer:
                    print(f"✓ transformer.{key}: {transformer[key]}")
                else:
                    print(f"✗ Missing required transformer key: {key}")
                    return False
            
            # Validate list lengths
            embed_dims = transformer['embed_dims']
            depths = transformer['depths']
            num_heads = transformer['num_heads']
            
            if len(embed_dims) == len(depths) == len(num_heads):
                print(f"✓ All transformer lists have same length: {len(embed_dims)}")
            else:
                print(f"✗ Transformer list lengths don't match: embed_dims={len(embed_dims)}, depths={len(depths)}, num_heads={len(num_heads)}")
                return False
                
        else:
            print("✗ Network configuration not found")
            return False
            
        # Check logging configuration
        if hasattr(config, 'logging'):
            print("✓ Logging configuration found")
            logging = config.logging
            print(f"  - enabled: {logging.get('enabled', 'Not set')}")
            print(f"  - write_to_file: {logging.get('write_to_file', 'Not set')}")
            print(f"  - write_to_console: {logging.get('write_to_console', 'Not set')}")
        else:
            print("✗ Logging configuration not found")
            return False
            
        print("\n✓ All configuration tests passed!")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_yaml_structure():
    """Test the YAML structure directly."""
    try:
        import yaml
        
        with open('config.yaml', 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        print("✓ YAML file loaded successfully")
        
        if 'network' in yaml_config:
            print("✓ Network section found in YAML")
            network = yaml_config['network']
            
            # Print network configuration
            print("\nNetwork Configuration:")
            for key, value in network.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for subkey, subvalue in value.items():
                        print(f"    {subkey}: {subvalue}")
                else:
                    print(f"  {key}: {value}")
        else:
            print("✗ Network section not found in YAML")
            return False
            
        if 'logging' in yaml_config:
            print("\nLogging Configuration:")
            logging = yaml_config['logging']
            for key, value in logging.items():
                print(f"  {key}: {value}")
        else:
            print("✗ Logging section not found in YAML")
            return False
            
        return True
        
    except ImportError:
        print("✗ PyYAML not available, skipping YAML structure test")
        return False
    except Exception as e:
        print(f"✗ YAML test error: {e}")
        return False

if __name__ == "__main__":
    print("Testing Configuration Structure")
    print("=" * 40)
    
    success1 = test_config_structure()
    print("\n" + "=" * 40)
    success2 = test_yaml_structure()
    
    if success1 and success2:
        print("\n🎉 All tests passed! Configuration is valid.")
    else:
        print("\n❌ Some tests failed. Please check the configuration.")
        sys.exit(1) 