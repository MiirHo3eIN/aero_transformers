import torch 
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt



from dataset_tr import TimeSeriesDataset
from model import WiBiTAD

folder_path = "~/project/Aerosense/data/aerosense_aerodynamic_data/aoa_0deg/"
test_experiments = [5,9, 24, 28, 43, 47, 62, 66, 81, 85, 100, 104]

seq_len = 70
d_model = 128 
n_heads = 8
def main_eval(): 

    test_x, test_y = TimeSeriesDataset(folder_path, test_experiments, seq_len = seq_len)    


    model = WiBiTAD(seq_len = seq_len, d_model= d_model, n_heads= n_heads)
    models_path = '../../models/00:01_tr.pt'
    model.load_state_dict(torch.load(models_path))    
    model.eval()
    
    test_x = (test_x[:, :1, :]).float()

    test_y = (test_y).detach().numpy()

    result = model(test_x).detach().numpy()
    print(test_y)   
    print(result)
if __name__ == "__main__":
    main_eval()