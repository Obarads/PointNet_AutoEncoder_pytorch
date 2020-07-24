import os, sys

sys.path.append("./")

import numpy as np
from plyfile import PlyData, PlyElement
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch

###
### Original:  https://github.com/loicland/superpoint_graph/
###
def write_ply(filename, xyz, rgb):
    """
    write into a ply file
    ref.:https://github.com/loicland/superpoint_graph/blob/ssp%2Bspg/partition/provider.py
    """
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop][0]] = xyz[:, i_prop]
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop+3][0]] = rgb[:, i_prop]
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)

try:
    from tsnecuda import TSNE
except:
    from sklearn.manifold import TSNE

def write_tsne(file_name, embedding, ins_label, sem_label=None):
    """
    base:https://medium.com/analytics-vidhya/super-fast-tsne-cuda-on-kaggle-b66dcdc4a5a4
    """
    x_embedding = TSNE().fit_transform(embedding)
    embedding_and_ins_label = pd.concat([pd.DataFrame(x_embedding), pd.DataFrame(data=ins_label,columns=["label"])], axis=1)
    sns.FacetGrid(embedding_and_ins_label, hue="label", height=6).map(plt.scatter, 0, 1).add_legend()
    plt.savefig(file_name)
    plt.close()
