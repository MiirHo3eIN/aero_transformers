import numpy as np 
import torch  
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import sys
import argparse

""" 
    This script implement the Feed Forward Layer from Attention is All You Need paper. 

    In the input we have a sequence of shape (batch_size, seq_len, d_model). 

    Then, during the feed forward layer we apply two linear transformations with a ReLU activation function in between them.

    The output is a tensor of shape (batch_size, seq_len, d_model)
"""


class PossitionWiseFeedForward(nn.Module): 
    def __init__(self, d_model: int, d_ff: int, drop_out: float):
        """ 
            Initialize the Possition Wise Feed Forward Layer from Attention is All You Need paper

            Parameters
            ----------
            d_model: int
                Dimension of the model
            d_ff: int
                Dimension of the feed forward layer
            drop_out: float
                Dropout rate
        """

        super().__init__()

        self._layer1 = nn.Linear(d_model, d_ff)
        self._layer2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """ 
            Propagate the input through the feed forward layer

            Parameters
            ----------
            x: torch.Tensor
                Input tensor of shape (batch_size, seq_len, d_model)
            
            Returns
            -------
            torch.Tensor
                Output tensor of shape (batch_size, seq_len, d_model)
        """

        x = self._layer1(x)
        x = F.relu(x)
        x = self._layer2(x)

        return x
    

def test_func(): 

    x = torch.randn((32, 70, 128))

    x_ff = PossitionWiseFeedForward(128, 512, 0.1)(x)


    print(x_ff.shape)

    plt.figure(figsize=(12, 8))
    plt.imshow(x[0].detach().numpy())
    
    plt.figure(figsize=(12, 8))
    plt.imshow(x_ff[0].detach().numpy())
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