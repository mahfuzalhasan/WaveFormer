# train_ssl_vit.py

import argparse
import os
from time import time

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from self_supervised.ssl_head import SSLViT
from loss import Loss
from scheduler import WarmupCosineSchedule
from data_utils import get_loader
from ops import augment_context_restoration


def main():
    def save_ckp(state, checkpoint_dir):
        torch.save(state, checkpoint_dir)

    def train(args, global_step, train_loader, val_best, scaler):
        model.train()
        loss_train = []
        loss_train_recon = []

        for step, batch in enumerate(train_loader):
            t1 = time()

            x = batch["image"].cuda()  # shape [B, C, H, W, D]
            # we typically want two augmented views for contrastive
            x1 = x.clone()
            x2 = x.clone()

            x1_augment = augment_context_restoration(x1)
            x2_augment = augment_context_restoration(x2)

            with autocast(enabled=args.amp):

                contrastive1_p, rec_x1 = model(x1_augment)
                contrastive2_p, rec_x2 = model(x2_augment)

                # combine for the loss function
                # e.g. for reconstruction, compare rec_x1 -> x1 and rec_x2 -> x2
                # for contrastive, compare contrastive1_p <-> contrastive2_p
                loss, losses_tasks = loss_function(
                    contrastive1_p,  # embeddings for aug1
                    contrastive2_p,  # embeddings for aug2
                    rec_x1,          # reconstruction for aug1
                    rec_x2,          # reconstruction for aug2
                    x1,              # original volume for x1
                    x2,              # original volume for x2
                )

            loss_train.append(loss.item())

            # Suppose losses_tasks returns [contrastive_loss, reconstruction_loss]
            # or a dict. Adjust indexing or keys as needed.
            # Example: losses_tasks[1] is reconstruction
            loss_train_recon.append(losses_tasks[1].item())

            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.lrdecay:
                scheduler.step()
            optimizer.zero_grad()

            # Logging
            if args.distributed:
                if dist.get_rank() == 0:
                    print(
                        f"Step:{global_step}/{args.num_steps}, "
                        f"Loss:{loss:.4f}, Time:{(time() - t1):.4f}"
                    )
            else:
                print(
                    f"Step:{global_step}/{args.num_steps}, "
                    f"Loss:{loss:.4f}, Time:{(time() - t1):.4f}"
                )

            global_step += 1

            # Evaluate every `args.eval_num` steps
            if args.distributed:
                val_cond = (dist.get_rank() == 0) and (global_step % args.eval_num == 0)
            else:
                val_cond = (global_step % args.eval_num == 0)

            if val_cond:
                val_loss, val_loss_recon, img_list = validation(args, test_loader)
                writer.add_scalar("Validation/loss_recon", val_loss_recon, global_step)
                writer.add_scalar("train/loss_total", np.mean(loss_train), global_step)
                writer.add_scalar("train/loss_recon", np.mean(loss_train_recon), global_step)

                # Write example images
                writer.add_image("Validation/x1_gt", img_list[0], global_step, dataformats="HW")
                writer.add_image("Validation/x1_aug", img_list[1], global_step, dataformats="HW")
                writer.add_image("Validation/x1_recon", img_list[2], global_step, dataformats="HW")

                if val_loss_recon < val_best:
                    val_best = val_loss_recon
                    checkpoint = {
                        "global_step": global_step,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                    save_ckp(checkpoint, os.path.join(logdir, "model_bestValRMSE.pt"))
                    print(f"Model saved! Best Recon. Val Loss: {val_best:.4f}, Current: {val_loss_recon:.4f}")
                else:
                    print(f"Model NOT saved. Best: {val_best:.4f}, Current: {val_loss_recon:.4f}")

        return global_step, loss, val_best

    def validation(args, test_loader):
        model.eval()
        loss_val = []
        loss_val_recon = []
        img_list = [None, None, None]

        with torch.no_grad():
            for step, batch in enumerate(test_loader):
                val_inputs = batch["image"].cuda()

                # same approach, 2 augmented versions
                x1 = val_inputs.clone()
                x2 = val_inputs.clone()
                x1_augment = augment_context_restoration(x1)
                x2_augment = augment_context_restoration(x2)

                with autocast(enabled=args.amp):
                    contrastive1_p, rec_x1 = model(x1_augment)
                    contrastive2_p, rec_x2 = model(x2_augment)
                    loss, losses_tasks = loss_function(
                        contrastive1_p,
                        contrastive2_p,
                        rec_x1,
                        rec_x2,
                        x1,
                        x2
                    )

                # Suppose losses_tasks = [contrastive_loss, reconstruction_loss]
                loss_recon = losses_tasks[1]
                loss_val.append(loss.item())
                loss_val_recon.append(loss_recon.item())

                # Grab an example slice for logging images
                x_gt = x1.detach().cpu().numpy()
                x_gt = (x_gt - np.min(x_gt)) / (np.max(x_gt) - np.min(x_gt) + 1e-8)
                xgt = (x_gt[0][0][:, :, 48] * 255.0).astype(np.uint8)

                x_aug = x1_augment.detach().cpu().numpy()
                x_aug = (x_aug - np.min(x_aug)) / (np.max(x_aug) - np.min(x_aug) + 1e-8)
                xaug = (x_aug[0][0][:, :, 48] * 255.0).astype(np.uint8)

                rec = rec_x1.detach().cpu().numpy()
                rec = (rec - np.min(rec)) / (np.max(rec) - np.min(rec) + 1e-8)
                recon = (rec[0][0][:, :, 48] * 255.0).astype(np.uint8)

                img_list = [xgt, xaug, recon]
                print(
                    f"Val step:{step}, Loss:{loss:.4f}, ReconLoss:{loss_recon:.4f}"
                )

        return np.mean(loss_val), np.mean(loss_val_recon), img_list

    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--logdir", default="test", type=str, help="directory to save logs")
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--num_steps", default=100000, type=int)
    parser.add_argument("--eval_num", default=100, type=int)
    parser.add_argument("--warmup_steps", default=500, type=int)
    parser.add_argument("--in_channels", default=1, type=int)
    parser.add_argument("--feature_size", default=48, type=int)
    parser.add_argument("--dropout_path_rate", default=0.0, type=float)
    parser.add_argument("--use_checkpoint", action="store_true")
    parser.add_argument("--spatial_dims", default=3, type=int)
    parser.add_argument("--a_min", default=-1000, type=float)
    parser.add_argument("--a_max", default=1000, type=float)
    parser.add_argument("--b_min", default=0.0, type=float)
    parser.add_argument("--b_max", default=1.0, type=float)
    parser.add_argument("--space_x", default=1.5, type=float)
    parser.add_argument("--space_y", default=1.5, type=float)
    parser.add_argument("--space_z", default=2.0, type=float)
    parser.add_argument("--roi_x", default=96, type=int)
    parser.add_argument("--roi_y", default=96, type=int)
    parser.add_argument("--roi_z", default=96, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--sw_batch_size", default=2, type=int)
    parser.add_argument("--lr", default=4e-4, type=float)
    parser.add_argument("--decay", default=0.1, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--lrdecay", action="store_true")
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--loss_type", default="SSL", type=str)
    parser.add_argument("--opt", default="adamw", type=str)
    parser.add_argument("--lr_schedule", default="warmup_cosine", type=str)
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--grad_clip", action="store_true")
    parser.add_argument("--noamp", action="store_true")
    parser.add_argument("--dist-url", default="env://")
    parser.add_argument("--smartcache_dataset", action="store_true")
    parser.add_argument("--cache_dataset", action="store_true")

    args = parser.parse_args()
    logdir = "./runs/" + args.logdir
    args.amp = not args.noamp
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)
    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
    args.device = "cuda:0"
    args.world_size = 1
    args.rank = 0

    if args.distributed:
        args.device = f"cuda:{args.local_rank}"
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method=args.dist_url)
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        print(f"Training in distributed mode. Process {args.rank}, total {args.world_size}.")
    else:
        print("Training with a single process on 1 GPU.")
    assert args.rank >= 0

    if args.rank == 0:
        os.makedirs(logdir, exist_ok=True)
        writer = SummaryWriter(logdir)
    else:
        writer = None

    model = SSLViT(args, upsample_mode="vae", hidden_size=512, projection_size=256)
    model.cuda()

    if args.opt == "adam":
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.decay)
    elif args.opt == "adamw":
        optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.decay)
    elif args.opt == "sgd":
        optimizer = optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.opt}")

    if args.resume:
        model_pth = args.resume
        model_dict = torch.load(model_pth)
        model.load_state_dict(model_dict["state_dict"])
        model.epoch = model_dict["epoch"]
        model.optimizer = model_dict["optimizer"]

    if args.lrdecay:
        if args.lr_schedule == "warmup_cosine":
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.num_steps)
        elif args.lr_schedule == "poly":
            def lambdas(epoch):
                return (1 - float(epoch) / float(args.epochs)) ** 0.9
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)
    else:
        scheduler = None


    # e.g. loss_function(contrastive1, contrastive2, rec1, rec2, orig1, orig2)
    loss_function = Loss(args.batch_size * args.sw_batch_size, args)


    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[args.local_rank])


    train_loader, test_loader = get_loader(args)

    global_step = 0
    best_val = 1e8
    scaler = GradScaler() if args.amp else None

    while global_step < args.num_steps:
        global_step, loss, best_val = train(args, global_step, train_loader, best_val, scaler)

    checkpoint = {
        "epoch": args.epochs,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }

    if args.distributed:
        if dist.get_rank() == 0:
            torch.save(model.state_dict(), os.path.join(logdir, "final_model.pth"))
        dist.destroy_process_group()
    else:
        torch.save(model.state_dict(), os.path.join(logdir, "final_model.pth"))
    save_ckp(checkpoint, os.path.join(logdir, "model_final_epoch.pt"))


if __name__ == "__main__":
    main()
