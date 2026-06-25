"""
S3DIS Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
from .defaults import DefaultDataset
from .builder import DATASETS
import numpy as np
from pointcept.utils.cache import shared_dict
import torch


@DATASETS.register_module()
class S3DISDataset(DefaultDataset):
    def __init__(
    self,
    la_file=None,
    **kwargs,
    ):
        self.la = torch.load(la_file) if la_file is not None else None
        super().__init__(**kwargs)

        
    def get_data_name(self, idx):
        remain, room_name = os.path.split(self.data_list[idx % len(self.data_list)])
        remain, area_name = os.path.split(remain)
        return f"{area_name}-{room_name}"
    
    def get_data(self, idx):
        data_dict = super().get_data(idx)  
        if self.la:
            sampled_index = self.la[self.get_data_name(idx)]
            mask = np.ones_like(data_dict["segment"], dtype=bool)
            mask[sampled_index] = False

            data_dict["labeled_mask"] = ~mask
            
        return data_dict
