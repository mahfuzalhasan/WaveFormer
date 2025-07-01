#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Example script demonstrating how to use the logging configuration.
This shows different logging levels and how they appear in both file and console.
"""

import sys
import os

# Add the parent directory to the path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from utils.logger_setup import setup_logging, get_logger


def main():
    """Example function showing different logging levels."""
    # Setup logging based on config
    setup_logging(config)
    Logger = get_logger()
    
    # Log different levels
    Logger.debug("This is a debug message - only visible in file if log_level_file is 'debug'")
    Logger.info("This is an info message - visible in both file and console")
    Logger.warn("This is a warning message - important information")
    Logger.error("This is an error message - something went wrong")
    Logger.critical("This is a critical message - severe error")
    
    # Example of logging with variables
    epoch = 10
    loss = 0.1234
    dice_score = 0.8567
    
    Logger.info(f"Epoch {epoch}: Loss = {loss:.4f}, Dice Score = {dice_score:.4f}")
    
    # Example of conditional logging
    if loss < 0.1:
        Logger.info("Loss is below threshold - good progress!")
    else:
        Logger.warn("Loss is still high - consider adjusting learning rate")
    
    # Example of logging lists/dicts
    metrics = {
        'tc_dice': 0.85,
        'wt_dice': 0.78,
        'et_dice': 0.92,
        'mean_dice': 0.85
    }
    
    Logger.info(f"Validation metrics: {metrics}")
    
    # Example of logging with different levels based on performance
    mean_dice = metrics['mean_dice']
    if mean_dice > 0.9:
        Logger.info(f"Excellent performance! Mean dice: {mean_dice:.4f}")
    elif mean_dice > 0.8:
        Logger.info(f"Good performance! Mean dice: {mean_dice:.4f}")
    elif mean_dice > 0.7:
        Logger.warn(f"Moderate performance! Mean dice: {mean_dice:.4f}")
    else:
        Logger.error(f"Poor performance! Mean dice: {mean_dice:.4f}")


if __name__ == "__main__":
    main() 