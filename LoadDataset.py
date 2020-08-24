from torch.utils.data import Dataset
import scipy.io as sio
import scipy.interpolate as scinp
import torch
import numpy as np


class LoadDataset(Dataset):
    
    def __init__(self, path_to_mat_file, transform=None):
        
        self.vehicleData = sio.loadmat(path_to_mat_file);
        self.outputData = self.vehicleData['norm_outputData'];
        self.inputData = self.vehicleData['norm_inputData'];
        

    def __len__(self):
        return int(self.outputData.shape[1]);

    def __getitem__(self, idx):
        inputData = self.inputData[:,:,idx];
        interpolatFunc = scinp.interp1d(np.arange(1,11,1),inputData,axis=0,fill_value='extrapolate');
        interpolatedInputData = interpolatFunc(np.arange(0.1,10.1,0.1));
        
        outputData = self.outputData[:,idx];         
        return torch.from_numpy(interpolatedInputData),torch.from_numpy(outputData);
