import os, sys
sys.path.append("./")

import numpy as np

import torch 

from pointnet_autoencoder.models.PointNetAutoEncoder import PointNetAutoEncoder
from pointnet_autoencoder.utils.setting import PytorchTools

if __name__ == "__main__":
    # use_feat_stn: use feature transform 
    # num_points: number of object points
    device = "cuda"
    num_points = 2048
    model = PointNetAutoEncoder(use_feat_stn=True, num_points=num_points)
    model.to(device)

    batch_size = 32
    num_channels = 3 # xyz

    data_list = []
    for i in range(5):
        points = torch.rand((batch_size, num_channels, num_points), dtype=torch.float32)
        points = points.to(device, non_blocking=True)
        global_features, _, _ = model.encoder(points)
        data_list.append(PytorchTools.t2n(global_features)) # t2n: torch tensor to numpy

    data_list = np.concatenate(data_list, axis=0)

    # global_features.shape: (batch_size*5, global_feature dim)
    # 32 * 5 = 160
    print(data_list.shape, data_list)

