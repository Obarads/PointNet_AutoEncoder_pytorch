import torch

def feature_transform_regularizer(trans):
    d = trans.shape[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, 
                      dim=(1,2)))
    return loss