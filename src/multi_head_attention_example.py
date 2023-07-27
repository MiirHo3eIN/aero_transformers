import torch 
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import argparse

from positional_encoding import Positional_Encoding


def test_func(): 

    d_model = 128 
    n = 70

    x_input = torch.randn((32, n, d_model))

    x_input_encoded = Positional_Encoding(n, d_model)(x_input)
    plt.figure(figsize=(12, 8))
    plt.plot(x_input_encoded[0, :,  5].detach().numpy(), label = 'x_input_encoded')

    print("Positional Encoding Shape:")
    print(x_input_encoded.shape)

    # define the MHA layer 
    mha = nn.MultiheadAttention(d_model, 4, dropout=0.0, batch_first=True)

    # apply the MHA layer
    
    # Make the query, key, and value tensors of shape (batch_size, sequence_length, d_model)
    x_query = torch.nn.Linear(d_model, d_model, bias=False)(x_input_encoded)
    x_key   = torch.nn.Linear(d_model, d_model, bias = False)(x_input_encoded)
    x_value = torch.nn.Linear(d_model, d_model, bias=False)(x_input_encoded)

    print("Query - Key - Value Shape:")
    print(x_query.shape)

    x_mha, _ = mha(x_query, x_key, x_value)

    print("Multi Head Attention Shape:")
    print(x_mha.shape)
    plt.figure(figsize=(12, 8))
    plt.plot(x_mha[0, :,  5].detach().numpy(), label = 'x_mha')
    plt.legend()
    plt.show()


# Initialize the parser
parser = argparse.ArgumentParser(description='Test Multi Head Attention the Encoder block.')

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