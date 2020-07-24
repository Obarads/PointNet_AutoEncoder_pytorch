import os, sys
sys.path.append("./")

import numpy as np

from pointnet_autoencoder.dataset.ModelNet import ModelNet
from pointnet_autoencoder.utils.converter import write_ply

if __name__ == "__main__":
    dataset = ModelNet("data/modelnet40_normal_resampled", 2048)

    points, label = dataset[4000]

    rgb = np.full((points.shape[0], 3), 255, dtype=np.int32)
    xyz = points[:, :3]
    write_ply("examples/tests/conf_ply.ply", xyz, rgb)
