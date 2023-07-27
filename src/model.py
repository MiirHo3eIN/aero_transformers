import numpy as np 
import torch 
import torch.nn as nn
from torch.nn import functional as F


import matplotlib.pyplot as plt


from positional_encoding import Positional_Encoding
import dataset_tr as dataset_tr 
from transformer import Transformer

"""
WiBiTAD: WInd turBIne Transformer-based model for Anomaly Detection.
"""


class WiBiTAD(nn.Module): 

    def __init__(self, seq_len: int, d_model: int, n_heads:int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = "WiBiTAD"
        self.seq_len = seq_len 
        self.d_model = d_model
        self.n_heads = n_heads
        # 1. Transformer Backebone
        self._transformer = Transformer(batch_size = 1, seq_len = self.seq_len, d_model = self.d_model, drop_out = 0.1, n_heads = self.n_heads)
        # 2. Classification Head
        self._classification_head = nn.Linear(self.seq_len*self.d_model, 6)
        self._softmax = nn.Softmax()



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagate the input through the WiBiTAD model
        """
        # 1. Transformer Backbone
        x = self._transformer(x)
        # 2. Classification Head
        x = torch.flatten(x, start_dim=1)
        x = self._classification_head(x)
        # 3. Softmax
        x = self._softmax(x)

        return x

