import numpy as np

import torch
from torch import nn
from torch.autograd import Variable

from pointnet_autoencoder.models.modules.layer import MLP1D, Linear

class PointNetAutoEncoder(nn.Module):
    def __init__(self, use_feat_stn, num_points):
        super(PointNetAutoEncoder, self).__init__()
        self.encoder = PointNetExtractor(
            use_feat_stn=use_feat_stn
        )

        self.decoder_layers = nn.Sequential(
            Linear(1024, 1024),
            Linear(1024, 1024),
            nn.Linear(1024, num_points * 3),
        )

        self.num_points = num_points

    def decoder(self, x):
        batch_size, num_channels = x.shape
        x = self.decoder_layers(x)
        x = x.view(batch_size, -1, 3) # shape: batch_size, num_points, coordinate
        return x

    def forward(self, x):
        x, coord_trans, feat_trans = self.encoder(x)
        x = self.decoder(x)
        return x, coord_trans, feat_trans

# https://github.com/fxia22/pointnet.pytorch/blob/master/pointnet/model.py
class PointNetExtractor(nn.Module):
    def __init__(self, use_feat_stn=True, with_pointwise_feat=False):
        super(PointNetExtractor, self).__init__()
        self.coord_stn = STNkd(k=3)
        if use_feat_stn:
            self.feat_stn = STNkd(k=64)
        self.conv1 = MLP1D(3, 64)
        self.conv2 = MLP1D(64, 128)
        self.conv3 = MLP1D(128, 1024, act=None)

        self.use_feat_stn = use_feat_stn
        self.with_pointwise_feat = with_pointwise_feat

    def forward(self, x):
        # get number of points
        num_points = x.shape[2]

        # transpose xyz
        coord_trans = self.coord_stn(x)
        x = self.transpose(x, coord_trans)

        # conv
        x = self.conv1(x)

        # transpose features
        if self.use_feat_stn:
            feat_trans = self.feat_stn(x)
            x = self.transpose(x, feat_trans)
        else:
            feat_trans = None

        # get a global feature
        pointwise_feat = x
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        if self.with_pointwise_feat:
            x = x.view(-1, 1024, 1).repeat(1, 1, num_points)
            return torch.cat([x, pointwise_feat], 1), coord_trans, feat_trans
        else:
            return x, coord_trans, feat_trans

    def transpose(self, x, trans):
        x = torch.transpose(x, 1, 2)
        x = torch.bmm(x, trans)
        x = torch.transpose(x, 1, 2)
        return x

class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.encoder = nn.Sequential(
            MLP1D(k, 64),
            MLP1D(64, 128),
            MLP1D(128, 1024),
        )

        self.decoder = nn.Sequential(
            Linear(1024, 512),
            Linear(512, 256),
            nn.Linear(256, k*k)
        )

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.encoder(x)

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.decoder(x)

        iden = torch.Tensor(torch.from_numpy(np.eye(self.k).flatten().astype(
                            np.float32))).view(1,self.k*self.k).repeat(
                            batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)

        return x



