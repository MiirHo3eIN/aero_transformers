import numpy as np 
import pandas as pd 
import torch 
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from scipy import signal

import shutup 

shutup.please()

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


class MeanDataset(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - torch.mean(x, dim=0)
    
class HighPassFilter(nn.Module): 

    def __init__(self, sampling_frequency, cutoff_frequency, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._sampling_frequency = sampling_frequency
        self._cutoff_frequency = cutoff_frequency

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # Going from torch to numpy 
        x = x.detach().cpu().numpy()

        x = self.__high_pass_filter__(x, 
                                fs = self._sampling_frequency, 
                                cutoff_freq = self._cutoff_frequency,
                                order = 4)
        
        
        return torch.from_numpy(x).float() 

    def __high_pass_filter__(self, data: np.array, fs = 100, cutoff_freq = 0.5, order = 4): 
    
        # Design the filter
        b, a = signal.butter(order, cutoff_freq, fs=fs, btype='high', analog=False, output='ba')
        # Apply the high-pass filter to the input signal
        filtered_signal = signal.lfilter(b, a, data)
        return filtered_signal


class RawDataset: 

    def __init__(self, folder_path, experiments: np.array, seq_len: int) -> None:

        self._folder_path = folder_path 
        self._eperiments = experiments
        self._datasetlen = len(experiments)
        self._seq_len = seq_len 


        self._mean = MeanDataset()
        self._high_pass_filter = HighPassFilter(sampling_frequency= 100, cutoff_frequency= 0.5)
    

    def __len__(self) -> int:
        return self._datasetlen
    

    def __getitem__(self, idx: int) -> np.array:

         
        del_cells = [0, 23, 37]
        cols = np.arange(0, 41)
        use_cols_ = np.delete(cols, del_cells)
        columns_ = [f"sensor_{sense}" for sense in use_cols_]

        labels = torch.tensor([])

        experiment = self._eperiments[idx]
        exp_num_three_digit = str(experiment).zfill(3)
        df = pd.read_csv(f"{self._folder_path}aoa_0deg_Exp_{exp_num_three_digit}_aerosense/1_baros_p.csv", header= None, skiprows= 2500, usecols = use_cols_)
        df.columns = columns_


        torch_df = torch.tensor(df.values, dtype = torch.float32)
        #print(torch_df.shape)

        # Apply Mean 
        mean_ = self._mean(torch_df)

        # Apply High Pass Filter
        high_passed_ = self._high_pass_filter(mean_)

        nrows, ncolumns  = high_passed_.shape 
        dim = self._seq_len
        N0 = nrows //dim 
        final_tensor = high_passed_[:N0*dim, :].reshape(N0, dim, ncolumns).permute(1, 0, 2).permute(1, 2, 0) 
        

        # Add labels based on the experiment number from the excel file
        if experiment < 20:  # Healthy Condition 
            labels = [1, 0, 0, 0, 0, 0]
        elif experiment < 39 and experiment > 19 :  # Mass added to the Structure
            labels = [0, 0, 0, 0, 0, 1] 
        elif experiment < 58 and experiment > 39 :  # Crack 5mm in the Structure
            labels = [0, 1, 0, 0, 0, 0] 
        elif experiment < 77 and experiment > 57 :  # Crack 10mm in the Structure
            labels = [0, 0, 1, 0, 0, 0] 
        elif experiment < 96 and experiment > 76 :  # Crack 15mm in the Structure
            labels = [0, 0, 0, 1, 0, 0] 
        else:                                       # Crack 20mm in the Structure 
            labels = [0, 0, 0, 0, 1, 0] 
        
        labels = torch.tensor(np.array(labels).reshape(1, 6) , dtype = torch.float32)
        labels = torch.broadcast_to(labels, (final_tensor.shape[0], 6))

        return final_tensor, labels.requires_grad_(True) 




def TimeSeriesDataset(folder_path, experiments: np.array, seq_len:int) -> torch.Tensor:

    dataset = RawDataset(folder_path, experiments, seq_len= seq_len)  
    
    sample_acum = 0
    for tensor_num in np.arange(0, len(dataset)):
        
        tensor , label = dataset[tensor_num]

        # convert list to tensor 
        #label = torch.tensor(label, dtype = torch.float32)
        label = label.clone().detach().requires_grad_(True)
        # print(tensor.shape)


        if tensor_num == 0:
            tensor_labels = label 
            tensors_cat = tensor
        else:
            tensors_cat = torch.cat((tensors_cat, tensor), dim = 0)
            tensor_labels = torch.cat((tensor_labels, label), dim = 0)

        sample_acum += tensor.shape[0]

    assert tensors_cat.shape[0] ==  sample_acum , "Concatenation of tensors is not working" 
    
    return tensors_cat , tensor_labels


def func_test():

    folder_path = "~/project/Aerosense/data/aerosense_aerodynamic_data/aoa_0deg/"
    train_experiments = np.arange(1, 10)
    seq_len = 100
    train_x , train_y = TimeSeriesDataset(folder_path, train_experiments, seq_len = seq_len)

    print(train_x.shape)
    print(train_y.shape)



#if __name__ == "__main__":
#    func_test()