import torch
import torch.nn as nn

from AutoReconstruction import AutoReconstruction
from Module2 import Module2
from UnivarNonlinResNet import UnivarNonlinResNet
from Module3 import Module3

# CURRENTLY ASSUMES ALL SHARED LAYERS ARE FIRST THEN K-PARALLEL LAYERS

class FullModel(nn.Module):
    """Collect all the relevant layers and define the structure of the overall network"""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.p = args.p
        self.Q = args.Q
        self.lrr_rank = args.lrr_rank
        self.no_clusters = args.no_clusters
        self.shared_layers = args.shared_layers
        self.parallel_layers = args.parallel_layers
        self.module_2_position = args.module_2_position
        self.module_strings = {
            "AR" : AutoReconstruction,
            "mod2" : Module2,
            "uni-nl" : UnivarNonlinResNet,
            "mod3" : Module3
        }
        # starts off blank, then updated by calling 
        self.all_layers = self._assemble_model()
    
    # leading underscore because only intended for internal use when setting up the model
    def _assemble_model(self):
        all_layers = []

        # add all necessary layers to the model
        for i in range(len(self.shared_layers)):
            all_layers.append(self.module_strings[self.shared_layers[i]](self.args))

        for i in range(self.no_clusters):
            for j in range(len(self.parallel_layers)):
                all_layers.append(self.module_strings[self.parallel_layers[j]](self.args))

        all_layers = nn.ModuleList(all_layers)

        return all_layers


    def forward(self, inputs):
        
        all_cluster_outputs = []

        for i in range(self.no_clusters):
            prev_output = inputs
            for j in range(len(self.shared_layers)):
                current_output = self.all_layers[j](prev_output)
                prev_output = current_output
            for j in range(len(self.parallel_layers)):
                current_output = self.all_layers[len(self.shared_layers) + i*(len(self.parallel_layers)) + j](prev_output)
                prev_output = current_output
            all_cluster_outputs.append(current_output[:,:,None])

        overall_output = torch.cat(all_cluster_outputs, dim=2)
        return overall_output
    
    # code adapted from Huang et al, 2021
    def predict_relationship_inside(self):        
        fea_weight_all = []      
        for cluster_i in range(self.no_clusters):
            Left = self.all_layers[1 + cluster_i*len(self.parallel_layers)+self.module_2_position].Vk#.transpose(0,1)
            
            tmp = nn.functional.normalize(Left, p=2, dim=0, eps=1e-10)

            # m
            tmp = torch.sum(torch.abs(tmp), dim=1) #+ torch.sum(torch.abs(Right), dim=0)

            # m
            tmp = 1- nn.functional.normalize(tmp, p=2, dim=0, eps=1e-10)

            # m
            tmp = nn.functional.normalize(tmp, p=2, dim=0, eps=1e-10)
            fea_weight_all.append(tmp)

            #print(f"Left dimensions: {Left.shape}")
            #print(f"Left: {Left}")
            #print(f"fea_weight_All: {fea_weight_all}")
            
        return fea_weight_all
