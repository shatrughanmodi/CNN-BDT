from __future__ import unicode_literals, division
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from LoadDataset import LoadDataset;
from PCEModule import PCEModule
import time

model = PCEModule();
model = model.cuda();

data = LoadDataset(path_to_mat_file='../Norm_TrainingData.mat');

## create training and validation split 
split = int(0.7 * len(data))
index_list = list(range(len(data)))
train_idx, valid_idx = index_list[:split], index_list[split:]

## create sampler objects using SubsetRandomSampler
tr_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(valid_idx)

batch_size = 64;
## create iterator objects for train and valid datasets
trainloader = DataLoader(data, batch_size=batch_size, sampler=tr_sampler)
validloader = DataLoader(data, batch_size=batch_size, sampler=val_sampler)

loss_function = torch.nn.MSELoss();
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

train_error,valid_error = [],[];

for epoch in range(1, 3001): ## run the model for 3000 epochs
    start = time.time();
    train_loss, valid_loss = [], []
    ## training part 
    model.train()
    for inputData,target in trainloader:
        optimizer.zero_grad()
        ## 1. forward propagation
        
        inputData = inputData.cuda();
        
        target = target.cuda();
        output = model(inputData)
        
        ## 2. loss calculation
        loss = loss_function(output, target)
        
        ## 3. backward propagation
        loss.backward()
        
        ## 4. weight optimization
        optimizer.step()
        
        train_loss.append(loss.item())        
        
    ## evaluation part 
    model.eval()
    
    for inputData,target in validloader:
        inputData = inputData.cuda();
        target = target.cuda();
        output = model(inputData)
        loss = loss_function(output, target)
        valid_loss.append(loss.item())
    print ("Epoch:", epoch, "Training Loss: ", np.mean(train_loss), "Valid Loss: ", np.mean(valid_loss))
    train_error.append(np.mean(train_loss));
    valid_error.append(np.mean(valid_loss));
    
    if epoch%5 == 0:
        torch.save(model.state_dict(), "model_" + str(epoch) + ".pt");
        np.save("train_error.npy",train_error);
        np.save("valid_error.npy",valid_error);
        
    end = time.time();
    print("Time taken: ", end-start);
