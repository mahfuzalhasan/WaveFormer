import os
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer
from monai.utils import set_determinism
from monai.losses.dice import DiceLoss
from monai.losses import DiceCELoss
from network_models import Waveformer, create_waveformer
from light_training.dataloading.dataset import get_train_val_test_loader_from_train
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from light_training.utils.files_helper import save_new_model_and_delete_last
from config import config
from utils.logger_setup import setup_logging, get_logger
from utils.network_config import get_network_config

set_determinism(123)

# Setup logging
setup_logging(config)
Logger = get_logger()


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        Logger.info(f"Created directory: {path}")


class BraTSTrainer(Trainer):
    """Custom Trainer for WaveFormer segmentation."""
    def __init__(
        self,
        env_type: str,
        max_epochs: int,
        batch_size: int,
        device: str = "cpu",
        val_every: int = 1,
        num_gpus: int = 1,
        train_process: int = 12,
        logdir: str = "./logs/",
        master_ip: str = 'localhost',
        master_port: int = 17750,
        training_script: str = "train.py"
    ):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        
        # Get network configuration
        network_config = get_network_config(config.__dict__)
        Logger.info(f"Network configuration: {network_config}")
        
        # Initialize model with configuration
        self.model = self._create_model(network_config)
        self.logdir = logdir
        
        # Store configuration for later use
        self.network_config = network_config
        self.roi_size = config.roi_size
        self.window_infer = SlidingWindowInferer(roi_size=self.roi_size, sw_batch_size=1, overlap=0.5)
        self.augmentation = True
        self.patch_size = self.roi_size
        self.best_mean_dice = 0.0
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.train_process = train_process
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0001)
        self.scheduler_type = None
        self.dice_loss = DiceCELoss(to_onehot_y=True, softmax=True)
        
        Logger.info(f"WaveFormerTrainer initialized with ROI size: {self.roi_size}, out_classes: {network_config.out_channels}")

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
        
        Logger.info(f"Creating model with parameters: {model_kwargs}")
        
        return create_waveformer(model_kwargs)

    def training_step(self, batch):
        """Performs a single training step."""
        image, label = self.get_input(batch)
        pred = self.model(image)
        loss = self.dice_loss(pred, label)
        self.log("training_loss", loss, step=self.global_step)
        return loss

    @staticmethod
    def convert_labels(labels: torch.Tensor) -> torch.Tensor:
        """Converts segmentation labels to multi-class format (TC, WT, ET)."""
        result = [
            (labels == 1) | (labels == 3),
            (labels == 1) | (labels == 3) | (labels == 2),
            labels == 3
        ]
        return torch.cat(result, dim=1).float()

    @staticmethod
    def get_input(batch) -> tuple:
        """Extracts image and label from batch."""
        image = batch["data"]
        label = batch["seg"]
        return image, label

    @staticmethod
    def cal_metric(gt, pred, voxel_spacing=[1.0, 1.0, 1.0]):
        """Calculates dice metric for a single class."""
        if pred.sum() > 0 and gt.sum() > 0:
            d = dice(pred, gt)
            return np.array([d, 50])
        elif gt.sum() == 0 and pred.sum() == 0:
            return np.array([1.0, 50])
        else:
            return np.array([0.0, 50])

    def validation_step(self, batch):
        """Performs a single validation step."""
        image, label = self.get_input(batch)
        output = self.model(image)
        output = output.argmax(dim=1)
        output = output[:, None]
        output = self.convert_labels(output)
        label = self.convert_labels(label)
        output = output.cpu().numpy()
        target = label.cpu().numpy()
        dices = []
        for i in range(3):
            pred_c = output[:, i]
            target_c = target[:, i]
            cal_dice, _ = self.cal_metric(target_c, pred_c)
            dices.append(cal_dice)
        return dices

    def validation_end(self, val_outputs):
        """Aggregates validation results and handles model saving."""
        dices = val_outputs
        tc, wt, et = dices[0].mean(), dices[1].mean(), dices[2].mean()
        Logger.info(f"Validation dice scores - TC: {tc:.4f}, WT: {wt:.4f}, ET: {et:.4f}")
        mean_dice = (tc + wt + et) / 3
        self.log("tc", tc, step=self.epoch)
        self.log("wt", wt, step=self.epoch)
        self.log("et", et, step=self.epoch)
        self.log("mean_dice", mean_dice, step=self.epoch)
        Logger.info(f'Epoch {self.epoch} - Mean dice: {mean_dice:.4f}')
        
        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            Logger.info(f"New best model! Mean dice improved to: {mean_dice:.4f}")
            save_new_model_and_delete_last(
                self.model, self.optimizer, mean_dice, self.epoch,
                os.path.join(self.logdir, f"best_model_{mean_dice:.4f}.pth"),
                delete_symbol="best_model"
            )
        
        save_new_model_and_delete_last(
            self.model, self.optimizer, mean_dice, self.epoch,
            os.path.join(self.logdir, f"final_model_{mean_dice:.4f}.pth"),
            delete_symbol="final_model"
        )
        
        if (self.epoch + 1) % 100 == 0:
            save_state = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'dice_score': mean_dice,
                'network_config': self.network_config.config  # Save network configuration
            }
            if self.scheduler is not None:
                save_state['lr_scheduler'] = self.scheduler.state_dict()
            checkpoint_path = os.path.join(self.logdir, f"tmp_model_ep{self.epoch}_{mean_dice:.4f}.pth")
            torch.save(save_state, checkpoint_path)
            Logger.info(f"Checkpoint saved at epoch {self.epoch}: {checkpoint_path}")


def main():
    # Configuration from config.yaml
    logdir = os.path.join(config.logdir, config.model_name)
    ensure_dir(logdir)
    ensure_dir(config.data_list_path)
    
    Logger.info("Starting WaveFormer training setup...")
    Logger.info(f"Data directory: {config.data_dir}")
    Logger.info(f"Log directory: {logdir}")
    Logger.info(f"Model name: {config.model_name}")
    
    # Validate network configuration
    try:
        network_config = get_network_config(config.__dict__)
        Logger.info(f"Network configuration validated successfully: {network_config}")
    except ValueError as e:
        Logger.error(f"Network configuration validation failed: {e}")
        raise
    
    train_ds, val_ds, test_ds = get_train_val_test_loader_from_train(
        config.data_dir, config.data_list_path, config.split_path)
    
    Logger.info(f"Dataset loaded - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    trainer = BraTSTrainer(
        env_type=config.env,
        max_epochs=config.max_epoch,
        batch_size=config.batch_size,
        device=config.device,
        logdir=logdir,
        val_every=config.val_every,
        num_gpus=config.num_gpus,
        train_process=config.train_process,
        master_port=config.master_port,
        training_script=__file__
    )
    
    Logger.info("Starting training...")
    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
    Logger.info("Training completed!")


if __name__ == "__main__":
    main()
