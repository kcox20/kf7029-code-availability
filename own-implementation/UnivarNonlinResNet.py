import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

class UnivarNonlinResNet(nn.Module):
    """Pytorch module learning univariate nonlinearities (Huang et al, 2021)."""
    def __init__(self, args):
        super().__init__()
        self.p = args.p
        self.Q = args.Q
        self.m = args.m

        self.args = args

        # setting up layers
        module_layers = []
        module_layers.append(nn.Linear(in_features=1, out_features=self.p))
        for i in np.arange(1, self.Q):
            module_layers.append(nn.Linear(in_features=self.p, out_features=self.p))
        module_layers = nn.ModuleList(module_layers)
        self.module_layers = module_layers

        # ADDED TO REPRODUCE ORIGINAL RESULTS (based on huang 2023)
        if self.args.use_huang_2023_args:
            print("Triggered initializing from normal distribution")
            for i in range(len(self.module_layers)):
                nn.init.normal_(self.module_layers[i].weight, mean=0, std=1e-10)
                nn.init.normal_(self.module_layers[i].bias, mean=0, std=1e-10)



    def forward(self, inputs): 
        if self.args.talk:
            print(f"\tUnivarNonlineResNet started ({datetime.now() - self.args.train_start_time})")

        prev_layer_output = inputs
        module_output = []
        for i in range(len(self.module_layers)):
            layer_output = self.module_layers[i](prev_layer_output)
            layer_output = nn.functional.relu(prev_layer_output + layer_output)
            module_output.append(layer_output)
            prev_layer_output = layer_output

        module_output = torch.cat(module_output, dim=2)

        module_output = module_output.transpose(1,2)

        if self.args.talk:
            print(f"\tUnivarNonlinResNet returned ({datetime.now() - self.args.train_start_time})")
        return module_output