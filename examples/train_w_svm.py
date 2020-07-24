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
    PytorchTools, is_absolute, make_folders, )
from pointnet_autoencoder.utils.metrics import Meter

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

    # pre-trained model file name
    model_name = "model.pth.tar"
    
    # start training
    for epoch in range(cfg.start_epoch, cfg.epochs):
        print('Epoch {}/{}:'.format(epoch, cfg.epochs))

        # training
        train_loss, train_dist_loss = train(cfg, model, dataset["train"], 
                                            optimizer, criterion)
        scheduler.step()
        print('-> Train loss: {}, dist_loss: {}'.format(train_loss, train_dist_loss))

        # evaluation
        if (epoch+1) % cfg.test_epoch == 0:
            test_acc, test_dist_loss = eval(cfg, model, dataset["train"], 
                                       dataset["test"], criterion)
            print('-> Test accracy: {}, dist_loss: {}'.format(test_acc, test_dist_loss))
        else:
            test_acc = 0

        # logging
        log_dict = {
            "epoch": epoch,
            "train/loss": train_loss,
            "train/dist_loss": train_dist_loss,
            "test/acc": test_acc,
            "test/dist_loss": test_dist_loss
        }
        logger.update(log_dict)

        # save params and model
        if epoch % cfg.save_epoch == 0 or epoch==cfg.epochs-1:
            torch.save({
                'epoch': epoch + 1,
                'cfg': cfg,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, model_name)
            print("Saving model to " + os.path.join(oscwd, model_name))

        if math.isnan(train_loss):
            print("Train loss is nan.")
            break
    
    torch.save({
        'epoch': cfg.epoch,
        'cfg': cfg,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }, "trained_model.pth.tar")
    print("Saving trained model to " + os.path.join(oscwd, model_name))

    print("Finish training.")

# training function
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
    batch_dist_loss = Meter()

    for lidx, (inputs, targets) in enumerate(loader):
        optimizer.zero_grad()

        inputs = inputs.to(cfg.device, non_blocking=True)
        inputs = torch.transpose(inputs, 1, 2)[:, :3] # inputs.shape: Batch_size, num_channels, num_points

        # model processing
        outputs, coord_trans, feat_trans = model(inputs)

        # compute losses
        loss = 0
        # chamfer distance loss
        inputs = torch.transpose(inputs, 1, 2)
        dist1, dist2 = criterion["chamfer_distance"](inputs, outputs) # dist1 and dist2 shape: batch_size, num_points
        dist_loss = (torch.mean(dist1)) + (torch.mean(dist2))
        loss += dist_loss

        # STNk loss
        if cfg.use_feat_trans:
            loss += criterion["feature_transform_regularizer"](feat_trans) * 0.001

        # for logger
        batch_loss.update(loss.item())
        batch_dist_loss.update(dist_loss.item())

        loss.backward()
        optimizer.step()

    return batch_loss.compute(), batch_dist_loss.compute()

# evaluation function (validation)
def eval(cfg, model, train_dataset, test_dataset, criterion, publisher="test"):
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
            inputs = torch.transpose(inputs, 1, 2)[:, :3] # inputs.shape: Batch_size, num_channels, num_points

            # model encoder processing
            outputs, _, _ = model.encoder(inputs)

            # add a global feature to a list
            train_global_features.append(PytorchTools.t2n(outputs))

        train_global_features = np.concatenate(train_global_features, axis=0) # shape (num_train_data, 1024) 

    # get global features using a validation dataset
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.nworkers,
        pin_memory=True
    )
    test_loader = tqdm(test_loader, ncols=100, desc="get eval GF")
    test_global_features = []
    eval_labels = []
    loss_list = []
    with torch.no_grad():
        for lidx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(cfg.device, non_blocking=True)
            inputs = torch.transpose(inputs, 1, 2)[:, :3] # inputs.shape: Batch_size, num_channels, num_points

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
            test_global_features.append(PytorchTools.t2n(outputs))

            # get eval labels
            eval_labels.append(targets)

        test_global_features = np.concatenate(test_global_features, axis=0) # shape (num_eval_data, 1024)
        eval_labels = np.squeeze(np.concatenate(eval_labels, axis=0),axis=-1) # shape (num_data)
        loss_list = np.concatenate(loss_list, axis=0)

    # use one class classification
    classifier = OneClassSVM(kernel='rbf', nu=0.1, gamma='auto')
    classifier.fit(train_global_features)
    pred_labels = classifier.predict(test_global_features)

    # get training data label
    _, true_label = train_dataset[0]
    # convert eval labels other than true labels to -1
    eval_labels[eval_labels != true_label] = -1
    # convert true labels to 1
    eval_labels[eval_labels == true_label] = 1

    # get loss of true data
    dist_loss = np.mean(loss_list[eval_labels]).item()
    # get a accuracy
    acc = np.mean(pred_labels == eval_labels).item()

    return acc, dist_loss

if __name__ == "__main__":
    main()

