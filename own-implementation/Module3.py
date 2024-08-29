import torch
import torch.nn as nn
from datetime import datetime
import numpy as np

class Module3(nn.Module):
    """Define a pytorch Module containing densely connected layers taking the m x pQ output of Module2 (the message passing stage of the GCN)"""
    def __init__(self, args):
        super().__init__()

        # set up to take output from module 2 as input therefore assumes the number of input features is pQ
        self.input_features = args.p * args.Q

        self.layer_sizes = args.module_3_layer_sizes
        # still linear; get dense rather than convolution behaviour because previous version in univarnonlin was a 'trick' treating the 
        # features as single feature instances to do the convolution
        # runs the pQ-element vector for each feature for each instance through the dense layer/s
        # input features is the number of elements in the last dimension of the input, so m x pQ is right to learn feature representation of pQ?
        module_layers = []
        # takes pQ input features to first element of module_3_layer_sizes output features
        module_layers.append(nn.Linear(in_features=self.input_features, out_features=self.layer_sizes[0]))
        # just doesn't add any more layers if only one layer size given
        for i in np.arange(1, len(self.layer_sizes)):
            module_layers.append(nn.Linear(in_features=self.layer_sizes[i-1], out_features=self.layer_sizes[i]))
        module_layers = nn.ModuleList(module_layers)
        self.module_layers = module_layers

        self.args = args
        
    def forward(self, inputs):
        """Define structure of the Module3 layers
        
        Argument:
        inputs: input data for the layer, expected to be output of module 2"""
        if self.args.talk:
            print(f"\tModule3 started ({datetime.now() - self.args.train_start_time})")

        prev_layer_output=inputs

        for i in range(len(self.module_layers)):
            layer_output = self.module_layers[i](prev_layer_output)
            layer_output = nn.functional.tanh(layer_output)
            prev_layer_output = layer_output
        
        # final output dimensions: instances x m x 1 (size of final dense layer)
        #print(f"Dimensions of final Module3 output: {layer_output.shape}")

        if self.args.talk:
            print(f"\tModule3 returned ({datetime.now() - self.args.train_start_time})")

        return layer_output
