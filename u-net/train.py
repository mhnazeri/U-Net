from pathlib import Path
import gc

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from thop import profile, clever_format

from loss import TverskyCrossEntropyDiceWeightedLoss
from model.u_net import UNet
from model.data_loader import CarlaSeg
from utils import (
    get_conf,
    check_grad_norm,
    save_checkpoint,
    load_checkpoint,
    timeit,
    decode_segmap,
    plot_images,
    init_weights_normal,
    class_labels,
)


def main(cfg_path: str, resume: bool = False):
    """main process to train the network"""
    # load config file
    cfg = get_conf(cfg_path)
    # Initialize wandb
    print("Initializing the logger:")
    wandb.init(
        project="U-Net",
        name="U-Net Baseline",
        tags="baseline",
        config=cfg,
        resume=resume
    )
    # create dataloader
    print("Initializing the dataloader")
    dl_cfg = cfg.dataloader_params
    if dl_cfg.dataset == "carla":
        data_root = dl_cfg.carla_root
    elif dl_cfg.dataset == "cityscapes":
        data_root = dl_cfg.cityscapes_root
    else:
        raise ValueError("Unknown value for dataset, it should be 'carla' or 'cityscapes'")

    dataset = CarlaSeg(
        root=data_root,
        n_classes=dl_cfg.n_classes,
        img_size=(dl_cfg.img_size.h, dl_cfg.img_size.w),
        mode="train",
    )
    data = DataLoader(
        dataset,
        batch_size=dl_cfg.batch_size,
        num_workers=dl_cfg.num_workers,
        shuffle=True,
        pin_memory=True,
    )
    val_dataset = CarlaSeg(
        root=dl_cfg.root_dir,
        n_classes=dl_cfg.n_classes,
        img_size=(dl_cfg.img_size.h, dl_cfg.img_size.w),
        mode="val"
    )
    val_data = DataLoader(
        val_dataset,
        batch_size=dl_cfg.batch_size,
        num_workers=dl_cfg.num_workers,
        shuffle=False,
        pin_memory=True,
    )
    # create model and optimizer
    print("Creating the model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    unet = UNet(cfg.unet.in_features, cfg.unet.out_features)
    unet.apply(init_weights_normal)
    unet = unet.to(device)
    train_cfg = cfg.train_params
    if train_cfg.optimizer.lower() == "adam":
        adam_cfg = cfg.adam
        optimizer = optim.Adam(
            unet.parameters(),
            lr=adam_cfg.lr,
           amsgrad=adam_cfg.amsgrad,
           betas=(adam_cfg.beta1, adam_cfg.beta2),
           eps=adam_cfg.eps,
           weight_decay=adam_cfg.weight_decay
        )
    elif train_cfg.optimizer.lower() == "rmsprop":
        rms_cfg = cfg.rmsprop
        optimizer = optim.RMSprop(
            unet.parameters(),
            lr=rms_cfg.lr,
            centered=rms_cfg.centered,
            momentum=rms_cfg.momentum,
            alpha=rms_cfg.alpha,
            eps=rms_cfg.eps,
            weight_decay=rms_cfg.weight_decay
        )
    else:
        print("Only RMSprop and Adam are supported")

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max")
    criterion = TverskyCrossEntropyDiceWeightedLoss(cfg.unet.out_features, device)

    if resume:
        # load checkpoint
        print("Loading checkpoint")
        save_dir = cfg.directory.load
        checkpoint = load_checkpoint(save_dir, device)
        unet.load_state_dict(checkpoint["unet"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        epoch = checkpoint["epoch"]
        best = checkpoint["best"]
        e_loss = checkpoint["e_loss"]
        dice = checkpoint["dice"]
        print(f"Loading checkpoint was successful, start from epoch {epoch}"
              f" and loss {best}")
    else:
        epoch = 1
        best = 0
        e_loss = []
        checkpoint = {
            "epoch": epoch,
            "e_loss": e_loss,
            "dice": [],
            "unet": None,
            "optimizer": None,
            "lr_scheduler": None,
            "best": best,
        }
    wandb.watch(unet)
    print("Start training")
    while epoch <= train_cfg.epochs:
        running_loss = []
        for idx, (img, seg) in enumerate(data):
            unet.train()
            optimizer.zero_grad()
            # move data to device
            img = img.to(device)
            seg = seg.to(device)

            # forward, backward
            out = unet(img)
            loss = criterion(out, seg)
            loss.backward()
            # check grad norm for debugging
            grad_norm = check_grad_norm(unet)
            # update
            optimizer.step()

            running_loss.append(loss.item())
            print(
                f"Batch {idx}, train loss: {loss.item():.2f}"
                f"\t Grad_Norm: {grad_norm:.2f}"
            )
            wandb.log(
                {
                    "epoch": epoch,
                    "batch": idx,
                    "loss": loss.item(),
                    "GradNorm": grad_norm,
                }
            )

        # validate on val set
        val_loss, t = validate(unet, val_data, criterion, wandb, device)
        lr_scheduler.step(val_loss[1])
        # average loss for an epoch
        e_loss.append(np.mean(running_loss))
        print(
            f"Epoch {epoch}, train Loss: {e_loss[-1]:.2f} \t Val loss: {val_loss[0]:.2f}"
            f"\t Dice: {torch.mean(val_loss[1]):.2f} \t time: {t:.3f} seconds"
        )
        wandb.log(
            {
                "epoch": epoch,
                "loss": e_loss[-1],
                "val_loss": val_loss[0],
                "dice": val_loss[1],
                "time": t,
            }
        )

        if epoch % train_cfg.save_every == 0:
            checkpoint["epoch"] = epoch
            checkpoint["unet"] = unet.state_dict()
            checkpoint["optimizer"] = optimizer.state_dict()
            # checkpoint["lr_scheduler"] = lr_scheduler.state_dict()
            checkpoint["e_loss"] = e_loss
            checkpoint["dice"] = dice
            # if the current average class dice is the best
            if val_loss[1] > best:
                best = val_loss[1]
                checkpoint["best"] = best
                wandb.run.summary["best_accuracy"] = best
                save_checkpoint(checkpoint, True, cfg.directory.save, str(epoch))
            else:
                save_checkpoint(checkpoint, False, cfg.directory.save, str(epoch))

            # plot some results, using wandb
            out = [x.cpu().detach().numpy().argmax(axis=0) for x in out]
            wandb.log(
                {
                    "Predictions": [
                        wandb.Image(
                            img[i].cpu().detach().permute(1, 2, 0).numpy(),
                            masks={
                                "predictions": {
                                    "mask_data": out[i],
                                    "class_labels": class_labels,
                                },
                                "ground_truth": {
                                    "mask_data": seg[i]
                                    .cpu()
                                    .detach()
                                    .numpy(),
                                    "class_labels": class_labels,
                                },
                            },
                        )
                        for i in range(img.size(0) // 2)
                    ]
                }
            )
            # plot some results, using matplotlib.
            # First, return them to cpu and convert to numpy array
            # out = torch.cat([torch.from_numpy(
            #     decode_segmap(x.cpu()
            #                   .detach()
            #                   .numpy()
            #                   .argmax(axis=0), cfg.unet.in_features))
            #     for x in out], dim=0)
            #
            # plot = torch.cat([img, out.unsqueeze(0)], dim=0)
            # plot_images(plot, "U-Net output")

        gc.collect()
        epoch += 1

    macs, params = op_counter(unet)
    print(macs, params)
    wandb.log({"GFLOPS": macs[:-1], "#Params": params[:-1]})
    print("ALL Finished!")


@timeit
def validate(model: torch.nn.Module, data, criterion, logger, device):
    """validate the network"""
    model.eval()
    # load config file
    # cfg = get_conf(cfg_path)
    running_loss = []
    running_dice = []

    for idx, (img, seg) in enumerate(data):
        # move data to device
        img = img.to(device)
        seg = seg.to(device)

        # forward, backward
        with torch.no_grad():
            out = model(img)
            loss = criterion(out, seg)
            running_loss.append(loss.item())
            d_loss = TverskyCrossEntropyDiceWeightedLoss.dice(out, seg)
            running_dice.append(d_loss)

    out = [x.cpu().numpy().argmax(axis=0) for x in out]
    logger.log(
        {
            "Predictions": [
                wandb.Image(
                    img[i].cpu().permute(1, 2, 0).numpy(),
                    masks={
                        "predictions": {
                            "mask_data": out[i],
                            "class_labels": class_labels,
                        },
                        "ground_truth": {
                            "mask_data": seg[i].cpu().numpy(),
                            "class_labels": class_labels,
                        },
                    },
                )
                for i in range(img.size(0) // 2)
            ]
        }
    )
    # average loss for an epoch
    loss = np.mean(running_loss)
    running_dice = torch.stack(running_dice, dim=0)
    dice = torch.mean(running_dice, dim=0)

    return loss, dice


def op_counter(model):
    model.eval()
    _input = torch.randn(1, 3, 256, 256)
    macs, params = profile(model, inputs=(_input,))
    macs, params = clever_format([macs, params], "%.3f")
    return macs, params


if __name__ == "__main__":
    cfg_path = Path("./conf/config")
    main(cfg_path)
