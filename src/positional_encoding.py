import numpy as np 
import torch
import torch.nn as nn
from torch.nn import functional as F

import matplotlib.pyplot as plt


class Positional_Encoding(nn.Module):
    """
    This function will return positional encoding for a sequence of length seq_len and embedding size d_model.
    It will return a torch tensor of shape (seq_len, d_model)
    """
    
    def __init__(self, seq_len:int, d_model:int = 512, dropout:float = 0.1 ,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.name = "Positional Encoding"
        self._dropout = nn.Dropout(dropout)
        self._d_model = d_model
        self._seq_len = seq_len
        
        # Create a numpy array of shape (seq_len, d_model)
        pe = torch.zeros((seq_len, d_model))
        # Create a numpy array of shape (d_model, )
        pos = torch.arange(0, seq_len).unsqueeze(1) 
        # Compute the denominator of the positional encoding function
        denominator = torch.pow(10000, torch.arange(0, d_model, 2, dtype = torch.float32) / d_model)
        # Compute the positional encoding function
        pe[:, 0::2] = torch.sin(pos / denominator)
        pe[:, 1::2] = torch.cos( pos / denominator)

        pe = pe.unsqueeze(0)

        # To make the positional encoding not trainable by optimizer 
        # For a better explanation please refer to <https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723>
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagate the input through the Positional Encoding Layer
        
        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Add the positional encoding to the input
        x = x + self.pe[:, :x.size(1)]
        return self._dropout(x)


def example_test():
    print("Running tests...")

    # Test 1
    d_model = 128
    seq_len = np.linspace(0, 100, 11, dtype=int)
    
    

    fig1, axs = plt.subplots(5,2, figsize=(10, 10))
    fig2, ax2 = plt.subplots(5,2, figsize=(10, 10))
    plot_idx = 0
    for n in seq_len:
        
        x_input = torch.randn((30, n, d_model))

        x_input_encoded = Positional_Encoding(n, d_model)(x_input)
        
        if plot_idx < 5:
            axs[plot_idx, 0].imshow(x_input_encoded[0])
            axs[plot_idx, 0].set_title(f"seg_len {n} and d_model {d_model}")
            ax2[plot_idx, 0].imshow(x_input[0])
            ax2[plot_idx, 0].set_title(f"seg_len {n} and d_model {d_model}")

        elif plot_idx>=5 and plot_idx<10: 
            axs[plot_idx-5, 1].imshow(x_input_encoded[0])
            axs[plot_idx-5, 1].set_title(f"seg_len {n} and d_model {d_model}")
            ax2[plot_idx-5, 1].imshow(x_input[0])
            ax2[plot_idx-5, 1].set_title(f"seg_len {n} and d_model {d_model}")
        
        plot_idx += 1
        print(x_input_encoded.shape)
    fig1.suptitle("Positional Encoding Output")
    fig2.suptitle("Input to Positional Encoding")
    fig1.tight_layout()
    fig2.tight_layout()
    plt.show()


# Uncomment the following lines to run the tests
if __name__ == "__main__":
    example_test()



