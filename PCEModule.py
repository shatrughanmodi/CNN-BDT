from __future__ import unicode_literals, division
import itertools
import torch

class PCEModule(torch.nn.Module):
    """
    Summary
    -------
        PCEModule for Estimating the energy consumption of EV

        The module is a multi-channel CNN, which extracts features using multiple convolution and 
        pooling layers from different input parameters separately and then combine the extracted 
        features. Then the combined features are passed through fully connected layers and the 
        final energy estimate is obtained.   
    """
    
    def __init__(self, n_channels=7, n_classes=10, dropout_probability=0.2):
        super(PCEModule, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.dropout_probability = dropout_probability

        # Layers ----------------------------------------------
    	 # Convolution Branch 1 Layers
        self.all_conv_high = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=8, kernel_size=7, padding=3),
            torch.nn.ReLU(),
            torch.nn.AvgPool1d(2),

            torch.nn.Conv1d(in_channels=8, out_channels=4, kernel_size=7, padding=3),
            torch.nn.ReLU(),
            torch.nn.AvgPool1d(2),

            torch.nn.Conv1d(in_channels=4, out_channels=4, kernel_size=7, padding=3),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.dropout_probability),
            torch.nn.AvgPool1d(2)
        ) for joint in range(n_channels-2)])
    
		 # Convolution Branch 2 Layers
        self.all_conv_medium = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.AvgPool1d(2),

            torch.nn.Conv1d(in_channels=8, out_channels=4, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.AvgPool1d(2),

            torch.nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.dropout_probability),
            torch.nn.AvgPool1d(2)
        ) for joint in range(n_channels-2)])
    
		 # Convolution Branch 3 Layers
        self.all_conv_low = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AvgPool1d(2),
            
            torch.nn.Conv1d(in_channels=8, out_channels=4, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AvgPool1d(2),

            torch.nn.Conv1d(in_channels=4, out_channels=4, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.dropout_probability),
            torch.nn.AvgPool1d(2)
        ) for joint in range(n_channels-2)])

		 # Residual Branch Layers
        self.all_residual = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.AvgPool1d(2),
            torch.nn.AvgPool1d(2),
            torch.nn.AvgPool1d(2)
        ) for joint in range(n_channels-2)])
    
		 # Fully Connected Layers
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features= (12 * 13 * (self.n_channels-2))+2, out_features=100), 
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=100, out_features=n_classes)
        )

        # Initialization --------------------------------------
        # Xavier init
        for module in itertools.chain(self.all_conv_high, self.all_conv_medium, self.all_conv_low, self.all_residual):
            for layer in module:
                if layer.__class__.__name__ == "Conv1d":
                    torch.nn.init.xavier_uniform_(layer.weight, gain=torch.nn.init.calculate_gain('relu'))
                    torch.nn.init.constant_(layer.bias, 0.1)

        for layer in self.fc:
            if layer.__class__.__name__ == "Linear":
                torch.nn.init.xavier_uniform_(layer.weight, gain=torch.nn.init.calculate_gain('relu'))
                torch.nn.init.constant_(layer.bias, 0.1)

    def forward(self, input):
        """
        This function performs the actual computations of the network for a forward pass.
        """
        inputs_to_convolve = input[:,:,0:self.n_channels-2];
        inputs_to_fc = input[:,0,self.n_channels-2:self.n_channels];
        
        all_features = []

        for channel in range(0, self.n_channels-2):
            input_channel = inputs_to_convolve[:, :, channel]

            input_channel = input_channel.unsqueeze(1)
            high = self.all_conv_high[channel](input_channel)
            medium = self.all_conv_medium[channel](input_channel)
            low = self.all_conv_low[channel](input_channel)
            ap_residual = self.all_residual[channel](input_channel)

            output_channel = torch.cat([
                high,
                medium,
                low,
                ap_residual
            ], dim=1)
            all_features.append(output_channel)

        all_features = torch.cat(all_features, dim=1)
        
        all_features = all_features.view(-1, 12 * 13 * (self.n_channels-2))  

        all_features = torch.cat((all_features,inputs_to_fc), dim=1);
        output = self.fc(all_features)        
        return output