import os, sys

import warnings
from os.path import join as opj 
import yaml
import numpy as np
import random
import pathlib
import subprocess

import torch
from torch.utils.data import Subset

def save_args(args, odir):
    if type(args) != dict:
        args = vars(args)
    with open(opj(odir,"args.yaml"),mode="w") as f:
        f.write(yaml.dump(args))

def make_folders(odir):
    if not os.path.exists(odir):
        os.makedirs(odir)

def is_absolute(path:str)->bool:
    path_pl = pathlib.Path(path)
    return path_pl.is_absolute()

def get_git_commit_hash():
    cmd = "git rev-parse --short HEAD"
    hash_code = subprocess.check_output(cmd.split()).strip().decode('utf-8')
    return hash_code

class PytorchTools:
    def __init__(self):
        print("This class is for staticmethod.")
    
    @staticmethod
    def create_subset(dataset, subset):
        if type(subset) is int:
            subset_number_list = np.random.randint(0,len(dataset)-1,subset)
        elif type(subset) is list:
            subset_number_list = subset
        else:
            NotImplementedError()
        return Subset(dataset,subset_number_list), subset_number_list
    
    @staticmethod
    def set_seed(seed, cuda=True, consistency=False):
        """ Sets seeds in all frameworks"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cuda: 
            torch.cuda.manual_seed(seed)
        if cuda and torch.cuda.is_available() and not consistency:
            torch.backends.cudnn.enabled = True # use cuDNN
        else:
            torch.backends.cudnn.enabled = False

    @staticmethod
    def select_device(device_name):
        if type(device_name) is str:
            if device_name in ["cpu", "-1"]:
                device = "cpu"
            elif device_name in ["cuda", "gpu","0"]:
                device = "cuda"
            elif device_name in ["tpu"]:
                raise NotImplementedError()
            else:
                raise NotImplementedError("1 Unknow device: {}".format(device_name))
        elif type(device_name) is int:
            if device_name < 0:
                device = "cpu"
            elif device_name >= 0:
                device = "cuda"
            else:
                raise NotImplementedError("2 Unknow device: {}".format(device_name))
        else:
            raise NotImplementedError("0 Unknow device: {}".format(device_name))
        return device

    @staticmethod
    def fix_model(model):
        for param in model.parameters():
            param.requires_grad = False

    @staticmethod
    def load_data(path):
        print("-> loading data '{}'".format(path))
        # https://discuss.pytorch.org/t/out-of-memory-error-when-resume-training-even-though-my-gpu-is-empty/30757
        checkpoint = torch.load(path, map_location='cpu')
        return checkpoint

    @staticmethod
    def resume(checkpoint, model, optimizer, scheduler):
        """
        return: model, optimizer, scheduler
        """
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint["scheduler"])
        return model, optimizer, scheduler

    @staticmethod
    def t2n(torch_tensor):
        return torch_tensor.cpu().detach().numpy()

    @staticmethod
    def dict2tensorboard(log_dict, writer, step):
        for key in log_dict:
            writer.add_scalar(key, log_dict[key] ,step)
