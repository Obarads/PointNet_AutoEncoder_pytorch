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
  ```bash
  python examples/train.py dataset_root=data/modelnet40_normal_resampled/
  ```
  - `examples/train.py` args was written on `examples/configs/config.yaml`.
- evaluation
  ```bash
  python examples/eval.py dataset_root=data/modelnet40_normal_resampled/ resume=outputs/YYYY-MM-DD/HH-MM-SS/model.pth.tar
  ```
  - `examples/eval.py` args was written on `examples/configs/config.yaml`.
