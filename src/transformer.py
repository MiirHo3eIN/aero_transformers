import numpy as np 
import torch 
from torch import nn
import matplotlib.pyplot as plt
import argparse

# Import Custom Modules
from input_embedding import InputEmbedding
from positional_encoding import Positional_Encoding
from encoder import Encoder



class Transformer(nn.Module): 

    def __init__(self, batch_size, input_channels, seq_len, d_model, drop_out, n_heads, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.name = "Transformer"

        # 0. Define the parameters of the model
        self.input_channels = input_channels
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.drop_out = drop_out
        self.n_heads = n_heads

        # 1. Define the input embedding layer
        self._input_embedding = InputEmbedding(seq_len = self.seq_len, d_model = self.d_model, batch_norm = True)

        # 2. Define the positional encoding layer
        self._positional_encoding = Positional_Encoding(seq_len = self.seq_len, d_model = self.d_model)

        # 3. Define the encoder layer 
        self._encoder = Encoder(n_batch = self.batch_size, d_model = self.d_model, n_h = self.n_heads, drop_out = self.drop_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        """
        Propagate the input through the Encoder Layer
        """
        # 0. Define the input embedding layer
        x = self._input_embedding(x)

        # 1. Define the positional encoding layer
        x = self._positional_encoding(x)

        # 2. Define the encoder layer 
        x = self._encoder(x)

        return x


def test_func(): 

    x = torch.rand(1, 1, 80)

    model = Transformer(batch_size = 1, seq_len = 80, d_model = 128, drop_out = 0.1, n_heads = 8)

    y = model(x) 

    print(y.shape)

    plt.figure()

    plt.imshow(y[0].detach().numpy())

    plt.figure()

    plt.plot(y[0].detach().numpy()[:, 0], label = "Transformer", color = "red")
    plt.plot(x[0].detach().numpy()[0], label = "Input")
    
    plt.legend()
    plt.show()


# Initialize the parser
parser = argparse.ArgumentParser(description='Transformer model test.')

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