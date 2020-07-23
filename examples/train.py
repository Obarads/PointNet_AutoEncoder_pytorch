import os, sys
sys.path.append("./")

import hydra
import omegaconf
import random
import numpy as np
import yaml
import math
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from pointnet_autoencoder.utils.setting import (
    PytorchTools, is_absolute, make_folders)
from pointnet_autoencoder.utils.metrics import Meter

from build_env import load_training_env, create_training_env, modify_path

# CONFIG_PATH = os.path.join(os.getcwd(), "configs/pointnet_s3dis.yaml")
CONFIG_PATH = "configs/s3dis.yaml"

@hydra.main(config_path=CONFIG_PATH, strict=False)
def main(cfg: omegaconf.DictConfig) -> None:
    # save note
    print("Note: ",cfg.note)

    # manage dir paths
    cwd = hydra.utils.get_original_cwd()
    oscwd = os.getcwd()
    cfg = modify_path(cwd, cfg)

    # resume
    if cfg.resume != None:
        cfg, dataset, model, optimizer, scheduler, criterion, \
            logger = load_training_env(cfg)
    else:
        cfg, dataset, model, optimizer, scheduler, criterion, \
            logger = create_training_env(cfg)
        cfg.start_epoch = 0

    with SummaryWriter(cfg.odir) as writer:
        # for resume
        if cfg.resume is not None:
            for i in range(0, cfg.start_epoch):
                PytorchTools.dict2tensorboard(logger.status[i], writer, i)

        # pre-trained model file name
        model_name = "model.pth.tar"
        
        # start training
        for epoch in range(cfg.start_epoch, cfg.epochs):
            print('Epoch {}/{}:'.format(epoch, cfg.epochs))
            train_loss, (train_acc, train_iou, train_overall) = train(
                cfg, model, dataset["train"], optimizer, criterion)
            scheduler.step()

            print('-> Train loss: {}  accracy : {}'.format(train_loss, 
                  train_acc))
            if (epoch+1) % cfg.test_epoch == 0:
                test_loss, (test_acc, test_iou, test_overall) = eval(
                    cfg, model, dataset["test"], criterion)
                print('-> Test loss: {}  accracy : {}'.format(test_loss, 
                      test_acc))
            else:
                test_loss = 0
                test_acc = 0
                test_iou = 0
                test_overall = 0

            # logging
            log_dict = {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "train/iou": train_iou,
                "train/overall": train_overall,
                "test/loss": test_loss,
                "test/acc": test_acc,
                "test/iou": test_iou,
                "test/overall": test_overall
            }
            PytorchTools.dict2tensorboard(log_dict, writer, epoch)
            logger.update(log_dict)

            # save params and model
            if epoch % cfg.save_epoch == 0 or epoch==cfg.epochs-1:
                torch.save({
                    'epoch': epoch + 1,
                    'cfg': cfg,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                },os.path.join(cfg.odir, model_name))
                print("Saving model to " + os.path.join(oscwd, cfg.odir, 
                      model_name))

            if math.isnan(train_loss):
                print("Train loss is nan.")
                break
        
        torch.save({
            'epoch': cfg.epoch,
            'cfg': cfg,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        },os.path.join(cfg.odir, "trained_model.pth.tar"))
        print("Saving trained model to " + os.path.join(oscwd, cfg.odir, 
              model_name))

        print("Finish training.")

# preprocessing of data
def preprocessing(cfg, inputs, targets):
    inputs = inputs[:, :3].contiguous()
    inputs = inputs.to(cfg.device, non_blocking=True)
    targets = targets.to(cfg.device, non_blocking=True)
    return inputs, targets

# model forward processing
def processing(cfg, inputs, targets, model):
    pred_semantic_labels, coord_trans, feat_trans = model(inputs)
    return pred_semantic_labels, coord_trans, feat_trans

# compute losses with criterion
def compute_loss(cfg, pred_semantic_labels, feat_trans, targets, criterion):
    loss = criterion["cross_entropy"](pred_semantic_labels, targets)
    if cfg.use_trans:
        loss += criterion["feature_transform_regularizer"](
            feat_trans) * 0.001
    return loss

# record each value (acc, loss..etc.) for computing average.
def meter_update(pred_semantic_labels, targets, loss, semantic_meters, 
                 batch_loss):
    semantic_meters.update(pred_semantic_labels, targets)
    batch_loss.update(loss.item())

# training
def train(cfg, model, dataset, optimizer, criterion, publisher="train"):
    model.train()
    loader = DataLoader(
        #Subset(dataset["train"],range(320)),
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.nworkers,
        pin_memory=True,
        shuffle=True,
        drop_last=True
    )
    loader = tqdm(loader, ncols=100, desc=publisher)
    batch_loss = Meter()

    for lidx, (inputs, targets) in enumerate(loader):
        optimizer.zero_grad()

        inputs, targets = preprocessing(cfg, inputs, targets)
        pred_semantic_labels, _, feat_trans = processing(
            cfg, inputs, targets, model)
        loss = compute_loss(cfg, pred_semantic_labels, feat_trans, 
                            targets, criterion)
        meter_update(pred_semantic_labels, targets, loss, semantic_meters,
                     batch_loss)

        loss.backward()
        optimizer.step()

    return batch_loss.compute(), semantic_meters.compute()

# evaluation
def eval(cfg, model, dataset, criterion, publisher="test"):
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.nworkers,
        pin_memory=True
    )
    loader = tqdm(loader, ncols=100, desc=publisher)
    batch_loss = Meter()

    with torch.no_grad():
        for lidx, (inputs, targets) in enumerate(loader):
            inputs, targets = preprocessing(cfg, inputs, targets)
            pred_semantic_labels, _, feat_trans = processing(
                cfg, inputs, targets, model)
            loss = compute_loss(cfg, pred_semantic_labels, feat_trans, 
                                targets, criterion)
            meter_update(pred_semantic_labels, targets, loss, semantic_meters,
                         batch_loss)

    return batch_loss.compute(), semantic_meters.compute()

if __name__ == "__main__":
    main()

