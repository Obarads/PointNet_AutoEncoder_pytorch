## Requirements
- 1. Install python packages
- ```bash
  pip install hydra-core --upgrade --pre  
  pip install pytorch
  ```
- Note: It is tested with:
  - python 3.6.10
  - torch==1.2.0

## Download dataset
- Download ModelNet40
  ```bash
  python examples/download.py
  ```

## How to use
- training
  - train PointNet AutoEncoder and  `sklearn.svm.OneClassSVM`
    ```bash
    python examples/train_w_svm.py dataset_root=data/modelnet40_normal_resampled/
    ```
    - `examples/train_w_svm.py` args was written on `examples/configs/config.yaml`. 
- evaluation
  - evaluate PointNet AutoEncoder with `sklearn.svm.OneClassSVM`, T-SNE and `ply` data
    ```bash
    python examples/eval_w_svm.py dataset_root=data/modelnet40_normal_resampled/ resume=outputs/YYYY-MM-DD/HH-MM-SS/model.pth.tar
    ```
    - `examples/eval.py` args was written on `examples/configs/config.yaml`.


## references
- [yanx27. Pointnet_Pointnet2_pytorch. In Github repository, 2019. (url:https://github.com/yanx27/Pointnet_Pointnet2_pytorch) (access:2020/7/14)](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
- [fxia22. pointnet.pytorch. In Github repository, 2017. (url:https://github.com/fxia22/pointnet.pytorch) (access:2020/7/20)](https://github.com/fxia22/pointnet.pytorch)
- [charlesq34. pointnet-autoencoder. In Github repository, 2018. (url:https://github.com/charlesq34/pointnet-autoencoder/blob/master/part_dataset.py) (access:2020/7/25)](https://github.com/charlesq34/pointnet-autoencoder/blob/master/part_dataset.py)
- [chrdiller. pyTorchChamferDistance. In Gihub repository, 2019. (url:https://github.com/chrdiller/pyTorchChamferDistance) (access:2020/7/25)](https://github.com/chrdiller/pyTorchChamferDistance)