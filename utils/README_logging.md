# Logging Configuration for BraTS Training

This directory contains the logging configuration system for the BraTS training project. The system provides configurable logging that can output to both files and console, with different log levels for each.

## Files

- `logger_setup.py`: Main logging setup utility
- `logging_example.py`: Example script showing how to use the logging system
- `README_logging.md`: This documentation file

## Configuration

The logging configuration is defined in `config.yaml` under the `logging` section:

```yaml
logging:
  enabled: true # Enable/disable logging
  write_to_file: true # Write logs to file
  write_to_console: true # Write logs to console
  log_file: "./logs/training.log" # Path to log file
  log_level_file: "debug" # Log level for file (debug, info, warning, error, critical)
  log_level_console: "info" # Log level for console
  log_format: "%(asctime)s %(levelname)-7s [%(filename)s:%(lineno)d] %(message)s"
  rewrite_log: false # Clear log file on startup
```

## Usage

### Basic Setup

```python
from config import config
from utils.logger_setup import setup_logging, get_logger

# Setup logging based on config
setup_logging(config)
Logger = get_logger()

# Use the logger
Logger.info("Training started")
Logger.warn("Warning message")
Logger.error("Error occurred")
```

### Log Levels

The system supports the following log levels (in order of increasing severity):

1. **DEBUG**: Detailed information for debugging
2. **INFO**: General information about program execution
3. **WARNING**: Warning messages for potentially problematic situations
4. **ERROR**: Error messages for serious problems
5. **CRITICAL**: Critical errors that may prevent the program from running

### Available Methods

- `Logger.debug(message)`: Log debug messages
- `Logger.info(message)`: Log info messages
- `Logger.warn(message)`: Log warning messages
- `Logger.error(message)`: Log error messages
- `Logger.critical(message)`: Log critical messages
- `Logger.info_once(message)`: Log info message only once (useful for repeated warnings)

### Configuration Options

#### Enable/Disable Logging

```yaml
logging:
  enabled: false # Disable logging completely
```

#### File vs Console Output

```yaml
logging:
  write_to_file: true # Write to file
  write_to_console: false # Don't write to console
```

#### Log Levels

```yaml
logging:
  log_level_file: "debug" # All messages to file
  log_level_console: "info" # Only info and above to console
```

#### Custom Log Format

```yaml
logging:
  log_format: "%(asctime)s [%(levelname)s] %(message)s"
```

#### Log File Management

```yaml
logging:
  log_file: "./logs/my_training.log" # Custom log file path
  rewrite_log: true # Clear log file on startup
```

## Examples

### Training Script Integration

```python
import os
from config import config
from utils.logger_setup import setup_logging, get_logger

# Setup logging at the start of your script
setup_logging(config)
Logger = get_logger()

def train_model():
    Logger.info("Starting model training...")

    for epoch in range(max_epochs):
        loss = train_epoch()
        Logger.info(f"Epoch {epoch}: Loss = {loss:.4f}")

        if epoch % 10 == 0:
            dice_score = validate_model()
            Logger.info(f"Validation Dice Score: {dice_score:.4f}")

            if dice_score > best_score:
                Logger.info(f"New best score: {dice_score:.4f}")
                save_model()

    Logger.info("Training completed!")

if __name__ == "__main__":
    train_model()
```

### Error Handling

```python
try:
    # Some operation that might fail
    result = risky_operation()
    Logger.info(f"Operation successful: {result}")
except Exception as e:
    Logger.error(f"Operation failed: {str(e)}")
    Logger.debug(f"Full error details: {e}")
```

### Performance Monitoring

```python
def log_metrics(epoch, metrics):
    """Log training metrics with appropriate levels."""
    mean_dice = metrics['mean_dice']

    if mean_dice > 0.9:
        Logger.info(f"Excellent performance! Mean dice: {mean_dice:.4f}")
    elif mean_dice > 0.8:
        Logger.info(f"Good performance! Mean dice: {mean_dice:.4f}")
    elif mean_dice > 0.7:
        Logger.warn(f"Moderate performance! Mean dice: {mean_dice:.4f}")
    else:
        Logger.error(f"Poor performance! Mean dice: {mean_dice:.4f}")
```

## Best Practices

1. **Setup logging early**: Call `setup_logging(config)` at the beginning of your main script
2. **Use appropriate log levels**:
   - DEBUG for detailed debugging information
   - INFO for general progress information
   - WARNING for potential issues
   - ERROR for actual errors
   - CRITICAL for severe problems
3. **Include context**: Log relevant information like epoch numbers, metrics, file paths, etc.
4. **Use structured logging**: Log dictionaries and objects for better readability
5. **Handle exceptions**: Always log exceptions with appropriate error levels

## Testing the Logging System

Run the example script to test the logging configuration:

```bash
python utils/logging_example.py
```

This will demonstrate different log levels and show how they appear in both the console and log file.

## Troubleshooting

### Logs not appearing in file

- Check that `write_to_file` is set to `true`
- Ensure the log directory exists and is writable
- Verify the `log_level_file` is set to an appropriate level

### Logs not appearing in console

- Check that `write_to_console` is set to `true`
- Verify the `log_level_console` is set to an appropriate level

### Permission errors

- Ensure the log directory has write permissions
- Check that the script has permission to create files in the specified directory
