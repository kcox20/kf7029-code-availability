import torch
import torch.nn as nn
from datetime import datetime


class AutoReconstruction(nn.Module):
    """Define a pytorch module to perform a simple, single layer reconstruction of the original features"""
    def __init__(self, args):
        super().__init__()
        # replaced to accomodate presence of module3 and options
        self.input_nonlins = args.input_nonlins

        self.m = args.m
        # initialised with 1e-10 (Huang et al, 2021)
        self.Rk_weight = nn.Parameter(1e-10 * torch.ones(self.m, self.input_nonlins))
        self.bias = nn.Parameter(1e-10 * torch.ones(self.m))

        self.args = args

    def forward(self, inputs):
        # [?]
        if self.args.talk:
            print(f"\tAutoReconstruction started ({datetime.now() - self.args.train_start_time})")
        temp_name = nn.functional.linear(input=inputs, weight=self.Rk_weight, bias=self.bias)
        all_instances = []
        for i in range(inputs.shape[0]):
            # dimensions: 819 (as expected?)
            current_instance = torch.diag(temp_name[i,:,:])
            #print(f"dimensions of current_instance : {current_instance.shape}")
            all_instances.append(current_instance[None, :])
        temp_name = torch.cat(all_instances, dim=0)
        #print(temp_name)
        #print(f"\nDimensions of final reconstriction (matmul version): {temp_name.shape}")
        #print(f"\nhuang autoreconstruction output: {final_y}")

        if self.args.talk:
            print(f"\tAutoReconstruction returned ({datetime.now() - self.args.train_start_time})")
        return temp_name