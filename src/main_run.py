import numpy as np 
import torch 
import torch.nn as nn
from torchinfo import summary


import plotly.graph_objects as go
import matplotlib.pyplot as plt


import dataset_tr as dataset_tr 
from model import WiBiTAD

from torch.utils.data import DataLoader, TensorDataset

import tqdm
from tqdm.notebook import tqdm_notebook

import random

import shutup 
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
shutup.please()

def generate_hexadecimal() -> str:
    hex_num  =  hex(random.randint(0, 16**16-1))[2:].upper().zfill(16)
    hex_num  =  hex_num[:4] + ":" + hex_num[4:8] + ":" + hex_num[8:12] + ":" + hex_num[12:]
    return hex_num



def main():

    folder_path = "~/project/Aerosense/data/aerosense_aerodynamic_data/aoa_0deg/"
    zeroing_experiments = [0, 1, 2, 6, 10, 11, 15, 19, 20, 25, 29, 30, 34, 38, 39,40, 44, 48,49, 53, 57, 58, 59, 63, 67,68, 72, 76, 77, 78, 82, 86, 87, 91, 95, 96, 97, 101, 105,106, 110] 
    test_experiments = [5,9, 24, 28, 43, 47, 62, 66, 81, 85, 100, 104]
    valid_experiments = [14, 18, 33, 37, 52, 56, 71, 75, 90, 94, 109, 113]
    train_experiments = np.delete(np.arange(0, 114), np.concatenate((test_experiments, valid_experiments, zeroing_experiments)))   

    print("Train experiments: ", train_experiments)
    print("Test experiments: ", test_experiments)
    print("Validation experiments: ", valid_experiments)
    
    seq_len = 70
    d_model = 128
    n_heads = 8


    train_x , train_y = dataset_tr.TimeSeriesDataset(folder_path, train_experiments, seq_len = seq_len)    
    # test_x, test_y = dataset_tr.TimeSeriesDataset(folder_path, test_experiments, seq_len = seq_len) 
    valid_x, valid_y = dataset_tr.TimeSeriesDataset(folder_path, valid_experiments, seq_len = seq_len)




    print(f"Train x shape: \t {train_x.shape} with {train_x.shape[0]} Training samples and {train_x.shape[1]} sequence length")
    print(f"Training labels samples: \t {train_y.shape[0]}")

    #print(f"Test x shape: \t {test_x.shape} with {test_x.shape[0]} Training samples and {test_x.shape[1]} sequence length")
    #print(f"Test labels samples: \t {test_y.shape[0]}")

    print(f"Validation x shape: \t {valid_x.shape} with {valid_x.shape[0]} Training samples and {valid_x.shape[1]} sequence length")
    print(f"Validation labels samples: \t {valid_y.shape[0]}")

    model = WiBiTAD(seq_len = seq_len, d_model= d_model, n_heads= n_heads)
    optimizer = torch.optim.Adam(model.parameters(), lr= 0.0001, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss()
    
    (summary(model, input_size = (1, 1, seq_len)))
    exit()
    dataset_train = TensorDataset(train_x[:, :1, :], train_y)
    train_loader = DataLoader(dataset_train, batch_size = 256, shuffle = False)

    dataset_valid = TensorDataset(valid_x[:, :1, :], valid_y)
    valid_loader = DataLoader(dataset_valid, batch_size = 256, shuffle = False)


    train_epoch_loss, valid_epoch_loss = [], []
    train_batch_loss, valid_batch_loss = [], []
    train_total_loss, valid_total_loss = [], []
    model_number = generate_hexadecimal()
    epochs = 100

    for epoch in np.arange(0, epochs):
        
        print(f"Epoch: {epoch+1}/{epochs}")

        ### TRAINING PHASE ###

        for x_batch, y_batch in (train_loader):
            #print("x_batch shape: ", x_batch.shape)
            #print("y_batch shape: ", y_batch.float().shape)
            
           
            y_train = model.forward(x_batch.float())
            train_loss = criterion(y_train.float(), y_batch.float())

            train_batch_loss += [train_loss]
            train_epoch_loss += [train_loss.item()]

            # Backpropagation
            optimizer.zero_grad()
            train_loss.backward()

            # Update the parameters
            optimizer.step()

        
        ### VALIDATION PHASE ###
        
        for x_batch, y_batch in (valid_loader):
        
            with torch.no_grad():
                model.eval()

                y_valid = model.forward(x_batch.float())
                valid_loss = criterion(y_valid.float(), y_batch.float())

                valid_batch_loss += [valid_loss]
                valid_epoch_loss += [valid_loss.item()]

        # Save the model 
        model_number = "00:01"
        torch.save(model.state_dict(), f"../../models/{model_number}_tr.pt")

        print(f"epoch{epoch+1}, \t Train loss = {sum(train_epoch_loss)/len(train_epoch_loss)}, \
        Validation Loss = {sum(valid_epoch_loss)/len(valid_epoch_loss)}")

        train_total_loss.append(sum(train_epoch_loss)/len(train_epoch_loss))
        valid_total_loss.append(sum(valid_epoch_loss)/len(valid_epoch_loss))

        train_epoch_loss, valid_epoch_loss = [], []
        train_batch_loss, valid_batch_loss = [], []
    

   

if __name__ == "__main__":
    
    
    print("Enter main")
    main()