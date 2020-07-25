import os, sys
sys.path.append("./")

import hydra
import omegaconf
import random
import numpy as np
import yaml
import math
from tqdm import tqdm
from sklearn.svm import OneClassSVM

import torch
from torch.utils.data import DataLoader

from pointnet_autoencoder.utils.setting import (
    PytorchTools, is_absolute, make_folders)
from pointnet_autoencoder.utils.metrics import Meter
from pointnet_autoencoder.utils.converter import write_ply, write_tsne

from build_env import load_training_env, create_training_env, modify_path

# CONFIG_PATH = os.path.join(os.getcwd(), "configs/pointnet_s3dis.yaml")
CONFIG_PATH = "configs/config.yaml"

@hydra.main(config_path=CONFIG_PATH, strict=False)
def main(cfg: omegaconf.DictConfig) -> None:
    # cfg: load configs/train.yaml

    # manage dir paths
    cwd = hydra.utils.get_original_cwd()
    oscwd = os.getcwd()
    cfg = modify_path(cwd, cfg)

    # resume training
    if cfg.resume != None:
        cfg, dataset, model, optimizer, scheduler, criterion, \
            logger = load_training_env(cfg)
    else:
        cfg, dataset, model, optimizer, scheduler, criterion, \
            logger = create_training_env(cfg)
        cfg.start_epoch = 0

    val_acc, val_dist_loss = eval(cfg, model, dataset["train"], 
                                dataset["test"], criterion)
    print('-> Test accracy: {}, dist_loss: {}'.format(val_acc, val_dist_loss))


# evaluation function
def eval(cfg, model, train_dataset, val_dataset, criterion, publisher="test"):
    model.eval()
    
    # get global features using a training dataset
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.nworkers,
        pin_memory=True
    )
    train_loader = tqdm(train_loader, ncols=100, desc="get train GF")
    train_global_features = []
    with torch.no_grad():
        for lidx, (inputs, targets) in enumerate(train_loader):

            inputs = inputs.to(cfg.device, non_blocking=True)
            inputs = torch.transpose(inputs, 1, 2)[:, :3] # inputs.shape: Batch_size, num_channels, num_points)
            # targets = targets.to(cfg.device, non_blocking=True)

            # model encoder processing
            outputs, _, _ = model.encoder(inputs)

            # add a global feature to a list
            train_global_features.append(PytorchTools.t2n(outputs))

        train_global_features = np.concatenate(train_global_features, axis=0) # shape (num_train_data, 1024) 

        # get reconstructions for ply data
        reconstructions = model.decoder(outputs)
        # save reconstructions as ply
        rgb = np.full((reconstructions.shape[1], 3), 255, dtype=np.int32)
        xyz = PytorchTools.t2n(reconstructions[0])
        write_ply("train_reconstruction.ply", xyz, rgb)
        inputs = torch.transpose(inputs, 1, 2)
        gt_xyz = PytorchTools.t2n(inputs[0])
        write_ply("train_input.ply", gt_xyz, rgb)

    # get global features using a eval dataset
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.nworkers,
        pin_memory=True
    )
    val_loader = tqdm(val_loader, ncols=100, desc="get eval GF")
    val_global_features = []
    eval_labels = []
    loss_list = []
    with torch.no_grad():
        for lidx, (inputs, targets) in enumerate(val_loader):

            inputs = inputs.to(cfg.device, non_blocking=True)
            inputs = torch.transpose(inputs, 1, 2)[:, :3] # inputs.shape: Batch_size, num_channels, num_points
            # targets = targets.to(cfg.device, non_blocking=True)

            # model encoder processing
            outputs, _, _ = model.encoder(inputs)

            # get reconstructions for loss of true data
            reconstructions = model.decoder(outputs)

            # compute loss
            inputs = torch.transpose(inputs, 1, 2)
            dist1, dist2 = criterion["chamfer_distance"](inputs, reconstructions)
            dist1 = np.mean(PytorchTools.t2n(dist1),axis=1)
            dist2 = np.mean(PytorchTools.t2n(dist2),axis=1)
            dist_loss = dist1 + dist2

            # add dist_losses to a list
            loss_list.append(dist_loss)

            # add a global feature to a list
            val_global_features.append(PytorchTools.t2n(outputs))

            # get eval labels
            eval_labels.append(targets)

        val_global_features = np.concatenate(val_global_features, axis=0) # shape (num_eval_data, 1024)
        eval_labels = np.squeeze(np.concatenate(eval_labels, axis=0),axis=-1) # shape (num_data)
        loss_list = np.concatenate(loss_list, axis=0)

        # save reconstructions as ply
        rgb = np.full((reconstructions.shape[1], 3), 255, dtype=np.int32)
        xyz = PytorchTools.t2n(reconstructions[0])
        write_ply("test_reconstruction.ply", xyz, rgb)
        gt_xyz = PytorchTools.t2n(inputs[0])
        write_ply("test_input.ply", gt_xyz, rgb)

    # use one class classification
    classifier = OneClassSVM(kernel='rbf', nu=0.1, gamma='auto')
    classifier.fit(train_global_features)
    pred_labels = classifier.predict(val_global_features)

    # visualize data using embeddings
    write_tsne("vis_embed.png", val_global_features, eval_labels)

    # get training data label
    _, true_label = train_dataset[0]
    # convert eval labels other than true labels to -1
    eval_labels[eval_labels != true_label] = -1
    # convert true labels to 1
    eval_labels[eval_labels == true_label] = 1

    # get loss of true data
    dist_loss = np.mean(loss_list[eval_labels])
    # get a accuracy
    acc = np.mean(pred_labels == eval_labels) * 100

    return acc, dist_loss

if __name__ == "__main__":
    main()

