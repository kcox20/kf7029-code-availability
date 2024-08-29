import torch
import torch.nn as nn
from datetime import datetime

# EXPECTS INPUT TO ALREADY HAVE BEEN TRANSPOSED so that it's ready to use directly
class Module2(nn.Module):
    """Define a pytorch Module to learn a low rank representation of a feature association matrix. Performs the message passing stage of a GCN.

    The feature association matrix is assumed to be symmetric (Huang et al, 2021)"""
    def __init__(self, args):
        super().__init__()
        # low rank representation rank (rank of LRR of feature association matrix)
        self.lrr_rank = args.lrr_rank
        self.Vk = nn.Parameter((1/args.m) * torch.ones(args.m, args.lrr_rank))
        #print(f"Original Vk: {self.Vk}")
        self.bias = nn.Parameter(1e-12 * torch.ones(args.m))

        self.args = args
        
    def forward(self, inputs):
        """Define structure of the Module2 layer[?]
        
        Argument:
        inputs: input data for the layer, expected to be univariate nonlinearities with dimensions pQ x m [!] [?] for each instance, b x pQ x m overall"""
        if self.args.talk:
            print(f"\tModule2 started ({datetime.now() - self.args.train_start_time})")
        self.Vk.data = nn.functional.normalize(self.Vk, eps=1e-10, dim=0)

        Ak = torch.matmul(self.Vk, torch.transpose(self.Vk, 0 , 1))
        Ak = Ak / torch.sum(Ak, dim=1, keepdim=True)

        Ak_zero_diag = torch.mul(Ak, (1 - torch.eye(self.Vk.shape[0])))

        layer_output = nn.functional.linear(input=inputs, weight=Ak_zero_diag, bias=self.bias)

        if self.args.talk:
            print(f"\tModule2 returned ({datetime.now() - self.args.train_start_time})")

        return nn.functional.tanh(layer_output.transpose(1,2))
