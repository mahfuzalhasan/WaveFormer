#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refactored prediction script for BraTS segmentation using Waveformer model.
This script follows Python best practices and uses the refactored network_models module.
"""

import os
import sys
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import external modules
from light_training.dataloading.dataset import get_train_val_test_loader_from_train
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.prediction import Predictor

# Import local modules
from config import config
from network_models import Waveformer, create_waveformer
from utils.logger_setup import setup_logging, get_logger
from utils.network_config import get_network_config

set_determinism(123)

def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing configuration file: {e}")


class BraTSPredictor(Trainer):
    """
    BraTS prediction trainer class.
    Handles model loading, prediction, and evaluation.
    """
    
    def __init__(self, config: dict, args: argparse.Namespace):
        """
        Initialize BraTS predictor.
        
        Args:
            config: Configuration dictionary
            args: Command line arguments
        """
        # Ensure log directory exists
        logdir = os.path.join(config['logdir'], config['model_name'])
        ensure_dir(logdir)
        ensure_dir(config['data_list_path'])

        # Setup logging
        setup_logging(config, train=False)
        self.logger = get_logger()
        
        # Initialize parent class
        super().__init__(
            env_type=config['env'],
            max_epochs=config['max_epoch'],
            batch_size=config['batch_size'],
            device=config['device'],
            val_every=config['val_every'],
            num_gpus=config['num_gpus'],
            logdir=logdir,
            master_port=config.get('master_port', 17751),
            training_script=__file__
        )
        
        # Store configuration
        self.config = config
        self.args = args
        self.patch_size = self.config['prediction']['patch_size']
        self.augmentation = False
        
        self.logger.info(f"BraTS Predictor initialized with device: {self.config['device']}")

        # self.model, self.predictor, self.save_path = self.define_model_waveformer()
    
    def convert_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Convert labels to TC, WT, and ET format.
        
        Args:
            labels: Input labels tensor
            
        Returns:
            torch.Tensor: Converted labels tensor
        """
        # TC, WT and ET
        result = [
            (labels == 1) | (labels == 3),  # TC: Tumor Core
            (labels == 1) | (labels == 3) | (labels == 2),  # WT: Whole Tumor
            labels == 3  # ET: Enhancing Tumor
        ]
        
        return torch.cat(result, dim=1).float()
    
    def _create_model(self, network_config) -> Waveformer:
        """
        Create model from network configuration.
        
        Args:
            network_config: Network configuration object
            
        Returns:
            Waveformer: Initialized model
        """
        # Get model keyword arguments
        model_kwargs = network_config.get_model_kwargs()
        
        # Add additional configuration
        model_kwargs['network_config'] = network_config.config
        
        self.logger.info(f"Creating model with parameters: {model_kwargs}")
        
        return create_waveformer(model_kwargs)
    
    def get_input(self, batch: dict) -> tuple:
        """
        Extract input data from batch.
        
        Args:
            batch: Input batch dictionary
            
        Returns:
            tuple: (image, label, properties)
        """
        image = batch["data"]
        label = batch["seg"]
        properties = batch["properties"]
        label = self.convert_labels(label)
        
        return image, label, properties
    
    def define_model_waveformer(self) -> tuple:
        """
        Define and load the Waveformer model.
        
        Returns:
            tuple: (model, predictor, save_path)
        """
        self.logger.info("Loading Waveformer model...")
        
        # Get network configuration
        network_config = get_network_config(self.config)
        
        # Create model using network configuration
        model = self._create_model(network_config)
        
        # Load model weights
        model_path = os.path.join(
            self.config['logdir'], 
            self.config['model_name'], 
            self.config['prediction']['best_model_id']
        )
        
        self.logger.info(f"Loading model from: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_dict = torch.load(model_path, map_location="cpu")
        new_sd = self._filter_state_dict(model_dict['model'])
        model.load_state_dict(new_sd, strict=True)
        model.eval()
        
        # Create sliding window inferer
        window_infer = SlidingWindowInferer(
            roi_size=self.config['prediction']['patch_size'],
            sw_batch_size=self.config['prediction']['sw_batch_size'],
            overlap=self.config['prediction']['overlap'],
            progress=True,
            mode="gaussian"
        )
        
        # Create predictor
        predictor = Predictor(
            window_infer=window_infer,
            mirror_axes=self.config['prediction']['mirror_axes']
        )
        
        # Create save path
        save_path = os.path.join(self.config['prediction']['prediction_save'], self.config['model_name'])
        os.makedirs(save_path, exist_ok=True)
        
        self.logger.info(f"Model loaded successfully. Save path: {save_path}")
        
        return model, predictor, save_path
    
    def validation_step(self, batch: dict, model: nn.Module, predictor: Predictor, save_path: str) -> int:
        """
        Perform validation/prediction step.
        
        Args:
            batch: Input batch
            model: Loaded model
            predictor: Prediction helper
            save_path: Path to save results
            
        Returns:
            int: Return code (0 for success)
        """
        image, label, properties = self.get_input(batch)
        
        # Perform prediction
        model_output = predictor.maybe_mirror_and_predict(image, model, device=self.config['device'])
        model_output = predictor.predict_raw_probability(model_output, properties=properties)
        
        # Convert to labels
        model_output = model_output.argmax(dim=0)[None]
        model_output = self.convert_labels_dim0(model_output)
        
        # Calculate dice scores
        label = label[0]
        num_classes = 3  # TC, WT, ET
        dices = []
        
        for i in range(num_classes):
            output_i = model_output[i].cpu().numpy()
            label_i = label[i].cpu().numpy()
            d = dice(output_i, label_i)
            dices.append(d)
        
        self.logger.info(f"Dice scores - TC: {dices[0]:.4f}, WT: {dices[1]:.4f}, ET: {dices[2]:.4f}")
        
        # Save prediction results
        model_output = predictor.predict_noncrop_probability(model_output, properties)
        predictor.save_to_nii(
            model_output,
            raw_spacing=self.config['prediction']['raw_spacing'],
            case_name=properties['name'][0],
            save_dir=save_path
        )
        
        return 0
    
    def convert_labels_dim0(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Convert labels to TC, WT, and ET format for dimension 0.
        
        Args:
            labels: Input labels tensor
            
        Returns:
            torch.Tensor: Converted labels tensor
        """
        # TC, WT and ET
        result = [
            (labels == 1) | (labels == 3),  # TC: Tumor Core
            (labels == 1) | (labels == 3) | (labels == 2),  # WT: Whole Tumor
            labels == 3  # ET: Enhancing Tumor
        ]
        
        return torch.cat(result, dim=0).float()
    
    def _filter_state_dict(self, state_dict: dict) -> dict:
        """
        Filter and clean state dictionary.
        
        Args:
            state_dict: Original state dictionary
            
        Returns:
            dict: Filtered state dictionary
        """
        if "module" in state_dict:
            state_dict = state_dict["module"]
        
        new_sd = {}
        for k, v in state_dict.items():
            k = str(k)
            new_k = k[7:] if k.startswith("module") else k
            new_sd[new_k] = v
        
        return new_sd


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="BraTS Prediction Script")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml",
        help="Path to configuration file"
    )
    # parser.add_argument(
    #     "--nautilus", 
    #     action='store_true', 
    #     help="Enable Nautilus environment"
    # )

    # user can use custom split path
    parser.add_argument(
        "--split-path", 
        type=str, 
        default="default_split",
        help="Data split path"
    )
    
    return parser.parse_args()


def main():
    """Main prediction function."""
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration --> user can load custom config file for prediction
    config = load_config(args.config)

    # Override split path if provided
    if args.split_path:
        config['split_path'] = args.split_path
    
    # Create predictor
    predictor = BraTSPredictor(config, args)
    
    # Load datasets
    train_ds, val_ds, test_ds = get_train_val_test_loader_from_train(
        config['data_dir'], 
        config['data_list_path'], 
        config['split_path'], 
        test=True
    )
    
    # Run prediction
    predictor.logger.info("Starting prediction on test dataset...")
    predictor.validation_single_gpu(test_ds)                        # batch size is 1 for prediction. set internally here.
    predictor.logger.info("Prediction completed successfully.")


if __name__ == "__main__":
    main() 