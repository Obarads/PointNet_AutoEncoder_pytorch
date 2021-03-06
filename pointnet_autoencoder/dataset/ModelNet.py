# ref: https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/data_utils/ModelNetDataLoader.py

import numpy as np
import warnings
import os
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def download(root):
    if not os.path.exists(root):
        os.mkdir(root)
    if not os.path.exists(os.path.join(root, "modelnet40_normal_resampled")):
        url = "https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip"
        # url = "https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip"
        zipfile = os.path.basename(url)
        os.system('wget %s --no-check-certificate; unzip %s' % (url, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], root))
        os.system('rm %s' % (zipfile))

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

class ModelNet(Dataset):
    def __init__(self, root,  npoint=1024, split='train', uniform=False, normal_channel=True, cache_size=15000, class_list=None):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel

        # select classes
        if class_list is not None:
            class_string = ",".join(class_list)
        else:
            class_string = ",".join(self.cat) # select all classes

        # get file_path list
        train_shape_ids = self.get_shape_ids('modelnet40_train.txt', class_string)
        test_shape_ids = self.get_shape_ids('modelnet40_test.txt', class_string)
        shape_ids = {"train": train_shape_ids, "test": test_shape_ids}

        # original code: https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/data_utils/ModelNetDataLoader.py#L50
        # shape_ids = {}
        # shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
        # shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        # confirm shape_ids when class_list=None
        # print(shape_ids["train"] == [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))])

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d'%(split,len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            # get data path
            fn = self.datapath[index]

            # get label
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32) # [cls]???

            # load points
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

            # save points to memory
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

        # select points
        if self.uniform:
            point_set = farthest_point_sample(point_set, self.npoints)
        else:
            choice = np.random.choice(len(point_set), self.npoints, replace=False)
            point_set = point_set[choice, :]

        # normalize coordinate
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        if not self.normal_channel:
            point_set = point_set[:, 0:3]

        # points_set.shape: (npoints, channels), cls: (1)
        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)
    
    def get_shape_ids(self, txt_file, class_string):
        shape_ids = []
        with open(os.path.join(self.root, txt_file)) as f:
            for line in f:
                obeject_file = line.replace("\n","")
                object_name = '_'.join(line.split('_')[0:-1])
                # Note: For example, when tv is selected, tv_stand is also included,
                #       but in ModelNet the characters do not overlap.
                if object_name in class_string: 
                    shape_ids.append(obeject_file)
        return shape_ids


if __name__ == '__main__':
    import torch
    data = ModelNet('/data/modelnet40_normal_resampled/',split='train', uniform=False, normal_channel=True,)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point,label in DataLoader:
        print(point.shape)
        print(label.shape)