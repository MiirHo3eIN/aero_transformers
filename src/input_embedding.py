import numpy as np 
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse

""" 
    This script deploy a 1D-CNN model to transform the input sequence into a vector of size d_model 

    In the input we have a sequence of shape (batch_size, seq_len, 1). By deplpying a 1D-CNN model we transform it to a tensor of shape (batch_size , seq_len, d_model)

    The idea is taken from A TRANSFORMER-BASED FRAMEWORK FOR MULTIVARIATE TIME SERIES REPRESENTATION LEARNING paper which is used for Time-series regression and classification tasks.  
    Link: https://arxiv.org/pdf/2010.02803.pdf
"""



class InputEmbedding(nn.Module): 

    def __init__(self,c_in:int , seq_len:int , d_model:int, *args, **kwargs) -> None:
        
        """
        Initialize the Input Embedding Layer from A TRANSFORMER-BASED FRAMEWORK FOR MULTIVARIATE TIME SERIES REPRESENTATION LEARNING paper

        Parameters
        ----------  
        c_in: int
            Number of input channels
        seq_len: int
            Length of the input sequence
        d_model: int
            Dimension of the model after the embedding layer
        """
        
        super().__init__(*args, **kwargs)
        self.name = "Input Embedding with 1D-CNN"

        self.c_in = c_in
        self._seq_len = seq_len
        self._d_model = d_model
        self._kernel_size = 3

        # Embedding layer
        self._conv1d = nn.Conv1d(self.c_in, self._d_model, self._kernel_size, padding= 1, stride=1, dilation=1, groups= 1)
        self._batch_norm = nn.BatchNorm1d(self._d_model, affine = False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagate the input through the Input Embedding Layer
        
        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (batch_size, 1, seq_len)
        
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        
        x = self._conv1d(x)
        x = self._batch_norm(x)
        x = x.permute(0, 2, 1)

        return x



def test_func(): 
    """
    Test the Input Embedding Layer

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    x = torch.randn((30 ,30, 70))
    input_embedding = InputEmbedding(30, 70, 128, True)
    output_tensor = input_embedding(x)
    print(output_tensor.shape)
    
    plt.figure()
    plt.plot(output_tensor[0].detach().numpy()[:, 0])
    plt.figure()
    plt.imshow(output_tensor[0].detach().numpy())
    plt.show()


# Create the parser to parse the arguments of the script for initiate testing


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

