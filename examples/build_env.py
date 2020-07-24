import os,sys
import yaml
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

# PointNet AutoEncoder Model
from pointnet_autoencoder.models.PointNetAutoEncoder import PointNetAutoEncoder

# loss
from pointnet_autoencoder.models.losses.chamfer_distance.chamfer_distance import \
    ChamferDistance

# regularizer function
from pointnet_autoencoder.models.losses.feature_transform_regularizer import \
    feature_transform_regularizer

# Dataset
from pointnet_autoencoder.dataset.ModelNet import ModelNet

# logger
from pointnet_autoencoder.utils.log_writer import LogWriter

# utils
from pointnet_autoencoder.utils.setting import (
    PytorchTools, is_absolute, make_folders)

### create optimizer
def create_optimizer(model, optimizer_name, lr, wd, betas=(0.9,0.999)):
    if optimizer_name == 'sgd':
        return optim.SGD(
            model.parameters(), 
            lr=lr,
            momentum=0.9,
            weight_decay=wd
        )
    elif optimizer_name == 'adam':
        return optim.Adam(
            model.parameters(), 
            lr=lr,
            betas=betas
        )

### dataset
def create_dataset(dataset, dataset_root, num_points, train_class_list, 
                   test_class_list):
    if dataset == "modelnet40":
        train_dataset = ModelNet(dataset_root, num_points, class_list=train_class_list)
        val_dataset = ModelNet(dataset_root, num_points, split="test",
                               class_list=test_class_list)
        dataset = {"train":train_dataset, "test":val_dataset}
    else:
        raise NotImplementedError('Unknown dataset ' + dataset)
    return dataset

def create_model(device, use_feat_trans, num_points):
    model = PointNetAutoEncoder(use_feat_stn=use_feat_trans, num_points=num_points)
    model.to(device)
    return model

def modify_path(cwd ,cfg):
    if not is_absolute(cfg.dataset_root):
        cfg.dataset_root = os.path.join(cwd, cfg.dataset_root)

    if cfg.resume is not None:
        if not is_absolute(cfg.resume):
            cfg.resume = os.path.join(cwd, cfg.resume)
    return cfg

def load_training_env(cfg):
    """
    inputs: cfg (omegaconf.DictConfig)

    return: cfg, dataset, model, optimizer, scheduler, calc_loss, logger
    """
    # save now params
    model_path = cfg.resume
    model_log_path = "/".join(cfg.resume.split("/")[:-1]) + \
        "/training_log.yaml"
    device = cfg.device
    if cfg.modify_epochs is not None:
        epochs = cfg.modify_epochs
    else:
        epochs = None

    # get checkpoint from model.pth.tar
    checkpoint = PytorchTools.load_data(model_path)

    # replace cfg with checkpoint cfg
    cfg = checkpoint["cfg"]

    # replace previous model params with now params or checkpoint
    cfg.device = device
    cfg.resume = model_path
    cfg.start_epoch = checkpoint["epoch"]
    if epochs is not None: cfg.epochs = epochs

    # build env
    cfg, dataset, model, optimizer, scheduler, calc_loss, \
        logger = create_training_env(cfg)
    model, optimizer, scheduler = PytorchTools.resume(
        checkpoint, model, optimizer, scheduler)

    # load logs
    logger.load_log(model_log_path)

    return cfg, dataset, model, optimizer, scheduler, calc_loss, logger

def create_training_env(cfg):
    """
    inputs: cfg (omegaconf.DictConfig)

    return: cfg, dataset, model, optimizer, scheduler, calc_loss, logger
    """
    
    # set seed 
    PytorchTools.set_seed(cfg.seed, cfg.cuda, cfg.reproducibility)

    # set device
    cfg.device = PytorchTools.select_device(cfg.device)

    # set start epochs
    cfg.start_epoch = 0

    # create dataset
    dataset = create_dataset(cfg.dataset, cfg.dataset_root, cfg.num_points, 
                             cfg.train_class_list, cfg.test_class_list)

    # if cfg.subset is not None, create subset dataset from dataset (for debug)
    if cfg.subset is not None:
        dataset["train"], subset_number_list = \
            PytorchTools.create_subset(dataset["train"],cfg.subset)
        dataset["test"], subset_number_list = \
            PytorchTools.create_subset(dataset["test"],cfg.subset)
        print("subset numbers:\n train {}, test {}".format(subset_number_list, 
              subset_number_list))

    print('Train dataset: {} elements - Test dataset: {} elements - '\
          'Validation dataset: {} elements'.format(len(dataset["train"]), 
          len(dataset["test"]), 0))

    # create model
    model = create_model(cfg.device, cfg.use_feat_trans, cfg.num_points)

    # create optimizer
    optimizer = create_optimizer(model, cfg.optim, cfg.lr, cfg.wd, cfg.betas)

    # create lr sheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.epoch_size, 
                                          gamma=cfg.gamma)

    # create loss func
    criterion = {}
    criterion["chamfer_distance"] = ChamferDistance()
    criterion["feature_transform_regularizer"] = feature_transform_regularizer

    # create writer
    logger = LogWriter(os.path.join("training_log.yaml"))

    return cfg, dataset, model, optimizer, scheduler, criterion, logger

