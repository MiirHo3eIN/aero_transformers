import numpy as np 
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import argparse

from positional_encoding import Positional_Encoding
from positionWiseFF import PossitionWiseFeedForward

class Encoder(nn.Module):
    
    def __init__(self,  n_batch: int, d_model:int, n_h :int,  drop_out: float):
        """
        
        Initialize the Encoder Layer from Attention is All You Need paper
        
        Parameters
        ----------
        d_model: int
            Dimension of the model 
        n_h: int
            Number of heads
        drop_out: float
            Dropout rate
        n_batch: int
            Batch size
        """
        
        
        super().__init__()
        self.name = "Transformer Encoder"


        # 0. Define the parameters of the model

        self.d_model = d_model
        self.n_h = n_h
        self.drop_out = drop_out
        self.n_batch = n_batch
        
        # 1. Define the key, query, and value tensors of shape (batch_size, d_model, d_model)
        self._x_query = torch.nn.Linear(d_model, d_model, bias=False) 
        self._x_key   = torch.nn.Linear(d_model, d_model, bias = False) 
        self._x_value = torch.nn.Linear(d_model, d_model, bias=False) 
        # 2. Define the self attention layer
        self._self_attention = nn.MultiheadAttention(embed_dim = d_model, num_heads = n_h, dropout = self.drop_out, batch_first= True)
        # 3. Add & Norm of MHA 
        self._layer_norm_1 = nn.LayerNorm(normalized_shape = self.d_model)
        # 4. Define the position-wise feed forward layer
        self._feed_forward = PossitionWiseFeedForward(d_model = self.d_model, d_ff = self.d_model*4, drop_out= self.drop_out)  # d_ff = 4*d_model from the Attention is All You Need paper
        # 5. Add & Norm of FF
        self._layer_norm_2 = nn.LayerNorm(normalized_shape = self.d_model)

    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        """
        Propagate the input through the Encoder Layer
        
        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        
        # input to self attention layer

        # Keep the residual of input for the skip connection
        residual = x

        # 1. Define the key, query, and value tensors of shape (batch_size, d_model, d_model)
        x_query = self._x_query(x)
        x_key   = self._x_key(x)
        x_value = self._x_value(x)
        # 2. Define the self attention layer
        x, _ = self._self_attention(x_query, x_key, x_value)
        # 3_A. Add of MHA
        x = x + residual
        # 3_B. Norm of MHA
        x = self._layer_norm_1(x)
        
        # 4_0 Keep the residual of MHA for the skip connection
        residual = x

        # 4. Define the position-wise feed forward layer 
        x = self._feed_forward(x)
        # 5_A. Add of FF
        x = x + residual
        # 5_B. Norm of FF
        x = self._layer_norm_2(x)
    
        return x        




def test_func():

    x = torch.randn((32, 70, 128))
    # Positional Encoding
    x_pos_encoded = Positional_Encoding(70, 128)(x)
    print(x_pos_encoded.shape)
    plt.figure(figsize=(12, 8))
    #plt.imshow(x_pos_encoded[0].detach().numpy(), cmap='viridis')
    #plt.colorbar()
    
    plt.plot(x_pos_encoded[0, :,  5].detach().numpy(), label = 'x_pos_encoded')
    

    # Encoder
    x_tran_encoded = Encoder(32, 128, 8, 0.0)(x_pos_encoded)


    print(x_tran_encoded.shape)

    #plt.figure(figsize=(12, 8))

    #plt.imshow(x_tran_encoded[0].detach().numpy(), cmap='viridis')
    #plt.colorbar()

    plt.plot(x_tran_encoded[0, :,  5].detach().numpy(), label = 'x_tran_encoded') 
    # plt.title('Transformer Encoding')
    plt.legend()
    plt.show()


# Initialize the parser
parser = argparse.ArgumentParser(description='Test Feed Forward layer of the Encoder block.')

# Add the parameters positional/optional
parser.add_argument('--test', type = bool, default = False, nargs='?', 
                    help='Test the script ')

# Parse the arguments
args = parser.parse_args()


if args.test == True:
    if __name__ == "__main__":
        test_func()
else:
    pass