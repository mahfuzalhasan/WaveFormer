#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Logging setup utility for the BraTS training project.
Uses the existing Logger class from lib.utils.tools.logger.
"""

import os
from lib.utils.tools.logger import Logger


def setup_logging(config):
    """
    Setup logging based on configuration.
    
    Args:
        config: Configuration object containing logging settings
    """
    if not hasattr(config, 'logging') or not config.logging.get('enabled', True):
        # If logging is disabled, only log to console with info level
        Logger.init(
            logfile_level=None,
            stdout_level='info',
            log_format='%(asctime)s %(levelname)-7s [%(filename)s:%(lineno)d] %(message)s'
        )
        return
    
    logging_config = config.logging
    
    # Determine log levels
    logfile_level = None
    stdout_level = None
    
    if logging_config.get('write_to_file', True):
        logfile_level = logging_config.get('log_level_file', 'debug')
    
    if logging_config.get('write_to_console', True):
        stdout_level = logging_config.get('log_level_console', 'info')
    
    # Get log file path
    log_file = logging_config.get('log_file', './logs/training.log')
    
    # Ensure log directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # Initialize logger
    Logger.init(
        logfile_level=logfile_level,
        stdout_level=stdout_level,
        log_file=log_file,
        log_format=logging_config.get('log_format', '%(asctime)s %(levelname)-7s [%(filename)s:%(lineno)d] %(message)s'),
        rewrite=logging_config.get('rewrite_log', False)
    )
    
    # Log the setup
    Logger.info(f"Logging initialized - File: {logfile_level}, Console: {stdout_level}, Log file: {log_file}")


def get_logger():
    """
    Get the configured logger instance.
    
    Returns:
        Logger: The configured logger instance
    """
    return Logger 