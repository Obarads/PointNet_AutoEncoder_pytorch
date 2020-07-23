import torch
import numpy as np
from scipy.stats import mode

class Meter:
    def __init__(self):
        self.v_list = []
    
    def update(self, v:float, coefficient:float=1):
        v = v / coefficient
        self.v_list.append(v)

    def compute(self):
        res = np.sum(self.v_list) / len(self.v_list)
        return res.tolist()
