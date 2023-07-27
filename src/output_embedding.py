import numpy as np 
import torch
import torch.nn as nn


class Output_Embedding(nn.Module):  

    def __init__(self, seq_len: int, d_model: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = "Output Embedding"

        self._seq_len = seq_len
        self._d_model = d_model

        # Embedding layer
        self._linear = nn.Linear(self._d_model, self._seq_len)