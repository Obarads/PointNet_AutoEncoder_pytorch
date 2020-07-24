import os, sys
sys.path.append("./")

import torch 
from pointnet_autoencoder.models.PointNetAutoEncoder import PointNetAutoEncoder

if __name__ == "__main__":
    # use_feat_stn: use feature transform 
    # num_points: number of object points
    device = "cuda"
    num_points = 2048
    model = PointNetAutoEncoder(use_feat_stn=True, num_points=num_points)
    model.to(device)

    batch_size = 32
    num_channels = 3 # xyz

    points = torch.rand((batch_size, num_channels, num_points), dtype=torch.float32)
    points = points.to(device, non_blocking=True)
    global_features, _, _ = model.encoder(points)

    # global_features.shape: (batch_size, global_feature dim)
    print(global_features.shape, global_features)
