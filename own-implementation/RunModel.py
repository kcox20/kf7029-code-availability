from FullModel import FullModel
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import scipy
from torch.autograd import Variable


class RunModel():
    """Perform interpretable deep clustering
    
    Arguments:
    * data_path: file path for input data (assumes each instance is a column in the file)
    * no_header_rows: number of header rows to be ignored when reading in the data
    * no_header_cols: number of header columns to be ignored when reading in the data
    * p: number of output units/features/filters in each ResNet layer in UnivarNonlinResNet
    * Q: number of ResNet layers in UnivarNonlinResNet
    * lrr_rank: rank of low rank representation learned for feature association matrix in Module2
    * no_clusters: number of clusters being learned (number of parallel networks)
    * lr: learning rate
    * shared_layers: list of strings corresponding to modules used (in order) in shared part of the network
        * "AR" : AutoReconstruction (final reconstruction, based on Huang et al, 2021)
        * "mod2" : Module2 (learning feature association matrix, based on Huang et al, 2021)
        * "uni-nl" : UnivarNonlinResNet (learning univariate nonlinearities, based on Huang et al, 2021)
    * parallel_layers: list of strings corresponding to modules used (in order) in each of the parallel networks (each parallel network corresponding to a cluster)
    """
    # [!] MISSING NEW ARGUMENTS
    def __init__(self,
                 data_path, 
                 no_header_rows=1,
                 no_header_cols=1,
                 delim=",",
                 p=10, 
                 Q=10, 
                 lrr_rank=1, 
                 no_clusters=2, # 7 for original methods data comparison
                 # key paper version: 1e-3
                 lr=1e-3, # changed from Huang et al 2021 (original had 1e-5)
                 weight_decay=5e-5, # from Huang et al 2021
                 shared_layers=["uni-nl"],
                 parallel_layers=["mod2", "mod3", "AR"],
                 module_3_layer_sizes=[10,1],
                 epochs=60, 
                 visualise_clusters=True,
                 cluster_vis_cols=3,
                 embedding="tsne", # should only be "pca" or "tsne"
                 batching=False,
                 batching_shuffle=False,
                 batch_size=100,
                 talk=False,
                 use_methods_data_instead=False,
                 use_huang_2023_args=False,
                 clusters_file_name="result/final_clusters.csv"
                 ):
        self.data_path = data_path
        self.no_header_cols = no_header_cols
        self.no_header_rows = no_header_rows
        self.delim = delim
        self.p = p
        self.Q = Q
        self.lrr_rank = lrr_rank
        self.no_clusters = no_clusters
        self.lr = lr
        self.weight_decay = weight_decay
        self.shared_layers = shared_layers
        self.parallel_layers = parallel_layers
        self.module_3_layer_sizes = module_3_layer_sizes
        self.epochs = epochs
        self.visualise_clusters = visualise_clusters
        self.cluster_vis_cols = cluster_vis_cols
        self.embedding = embedding
        self.batching = batching
        self.batching_shuffle = batching_shuffle
        self.batch_size = batch_size
        self.clusters_file_name = clusters_file_name

        self.use_huang_2023_args = use_huang_2023_args

        self.loss_metric = nn.MSELoss(size_average=True, reduce=False)
        
        # added to enable the module2 layer to be selected from the overall model for each cluster
        # (each module only counts as 1 in the list)
        if "mod2" in self.parallel_layers:
            self.module_2_position = self.parallel_layers.index("mod2")

        self.talk = talk
    
        data = pd.read_csv(self.data_path)

        if self.no_header_rows > 0:
            effective_header_rows = self.no_header_rows - 1
        else:
            effective_header_rows = 0
        data = data.iloc[effective_header_rows:, self.no_header_cols:]
        data = torch.from_numpy(data.to_numpy())
        data = torch.transpose(data, 0, 1)
        # version to compare output of model to
        data = data.float()
        # version to be used as input to model
        data_input = data[:,:,None]

        if batching:
            self.batched_index = []
        else:
            self.batched_index = torch.LongTensor(range(len(data_input)))

        if use_methods_data_instead:
            data_org = scipy.io.loadmat("C:/Users/katie/Downloads/FCAC/FCAC/data/segment_uni_norm.mat")['fea']
            data = data = torch.from_numpy(data_org).float()
            data_input = data[:,:,None]
            # override number of clusters to 7 (to match methods paper) if testing with methods data
            self.no_clusters = 7

        if use_huang_2023_args:
            self.lr = 1e-3
            self.weight_decay = 5e-5
            self.batching = True

        # calculated properties:
        # ASSUMES THE ONLY 2 OPTIONS ARE:
        # * NO MODULE 3, SO TAKING INPUT DIRECT FROM MODULE 2, SO pQ
        # * MODULE 3, SO SIZE OF LAST MODULE 3 LAYER
        if "mod3" in self.parallel_layers:
            self.input_nonlins = self.module_3_layer_sizes[-1]
        else:
            self.input_nonlins = self.p * self.Q

        self.m = data_input.shape[1]

        model = FullModel(self)
        print(model)
        # from huang et al 2021
        print(f"number of parameters: {sum([p.nelement() for p in model.parameters()])}")

        # hard coded:
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        #print(f"Dimensions of data_input (data_in): {data_input.shape}")
        #print(f"Dimensions of data (data_out): {data.shape}")

        self._perform_training(model=model, data_in=data_input, data_out=data)

    # code adapted directly from key paper (Huang et al, 2023)
    # default batch size set to 100 to match key paper
    #def get_batches(data, batch_size=100, shuffle = False):  
    def get_batches(self, in_data, out_data, batch_size=100, shuffle = False):  
        inputs = in_data
        targets = out_data
        length = len(inputs)
        
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        
        self.batched_index = index

        start_idx = 0
            
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]; 
            Y = targets[excerpt];   
            # from autograd           
            data = [Variable(X), Variable(Y)]
            yield data
            start_idx += batch_size

    # some original code from key paper with additions/changes
    def _train_loop(self, model, in_data, out_data):

        if self.batching:
            total_loss = 0
            residuals_weighted_agg = []
            # swapped to take in_data and out_data
            for inputs in self.get_batches(in_data, out_data, batch_size=self.batch_size, shuffle=self.batching_shuffle):
                X, Y = inputs[0], inputs[1]
                model.zero_grad()
                output = model(X) 

                if self.talk:
                    print(f"\tpredict_relationship_inside() started ({datetime.now() - self.train_start_time})")

                fea_scaler = model.predict_relationship_inside()

                if self.talk:
                    print(f"\tpredict_relationship_inside() finished ({datetime.now() - self.train_start_time})")

                residuals_weighted_agg_batch = torch.zeros((len(Y), output.shape[2]))

                for cluster_i in range(model.no_clusters):
                    # swapped to match own format
                    #residuals_tmp = self.loss_metric(output[cluster_i], Y)
                    residuals_tmp = self.loss_metric(output[:,:,cluster_i], Y)  ### raw residuals: bxm
                    #print(f"Dimensions of residuals_tmp: {residuals_tmp.shape}")
                    weight_tmp = fea_scaler[cluster_i][None, :].repeat(len(Y),1) ### weighting vector: bxm
                    #print(f"Dimensions of weight_tmp: {weight_tmp.shape}")
                    residuals_tmp = torch.mul(residuals_tmp, weight_tmp)  ## weighting
                    #print(f"Dimensions of residuals_tmp (after weighting): {residuals_tmp.shape}")
                    residuals_weighted_agg_batch[:, cluster_i] = torch.sum( residuals_tmp, dim=1)   ### sum_residual: b 
                    #print(f"Dimensions of residuals_weighted_agg_batch: {residuals_weighted_agg_batch.shape}")

                residuals_weighted_agg.append(residuals_weighted_agg_batch)
        
                loss_org = torch.sum(torch.min(residuals_weighted_agg_batch, dim=1)[0])   
                                    
                loss_org.backward(retain_graph=True)
                total_loss += loss_org.data.item()
                self.optimizer.step() 

            residuals_weighted_agg = (torch.cat(residuals_weighted_agg, dim = 0))
            cluster_preds = torch.min(residuals_weighted_agg, dim=1)[1]


                
        else:
            X = in_data
            Y = out_data

            model.zero_grad()

            output = model(X)

            if self.talk:
                print(f"\tpredict_relationship_inside() started ({datetime.now() - self.train_start_time})")

            fea_scaler = model.predict_relationship_inside()

            if self.talk:
                print(f"\tpredict_relationship_inside() finished ({datetime.now() - self.train_start_time})")

            residuals_weighted_agg = torch.zeros((X.shape[0], self.no_clusters))

            for cluster_i in range(model.no_clusters):
                if self.talk and cluster_i == 0:
                    print(f"\tcalculating MSE for cluster {cluster_i} ({datetime.now() - self.train_start_time})")
                residuals_tmp = self.loss_metric(output[:,:,cluster_i], Y)
                if self.talk and cluster_i == 0:
                    print(f"\tcalculating weighted loss for cluster {cluster_i} ({datetime.now() - self.train_start_time})")
                weight_tmp = fea_scaler[cluster_i][None, :].repeat(len(Y),1) ### weighting vector: bxm
                residuals_tmp = torch.mul(residuals_tmp, weight_tmp) ## weighting
                residuals_weighted_agg[:, cluster_i] = torch.sum( residuals_tmp, dim=1)   ### sum_residual: b 
                if self.talk and cluster_i == 0:
                    print(f"\tfinished calculating weighted loss for cluster {cluster_i} ({datetime.now() - self.train_start_time})")
    
            loss_org = torch.sum(torch.min(residuals_weighted_agg, dim=1)[0])   
                        
            if self.talk:
                print(f"\tloss_org.backward started ({datetime.now() - self.train_start_time})")
            loss_org.backward(retain_graph=True)
            if self.talk:
                print(f"\tloss_org.backward finished ({datetime.now() - self.train_start_time})")
            
            total_loss = loss_org.data.item()

            if self.talk:
                print(f"\toptimizer.step started ({datetime.now() - self.train_start_time})")
            self.optimizer.step()   
            if self.talk:
                print(f"\toptimizer.step finished ({datetime.now() - self.train_start_time})")

            cluster_preds = torch.min(residuals_weighted_agg, dim=1)[1]

        return_loss = total_loss / (out_data.shape[0] * out_data.shape[1])
        
        print(f"total_loss = {total_loss}")
        
        return return_loss, cluster_preds


    def _test_loop(self):
        print("UNFINISHED")

    def _perform_training(self, model, data_in, data_out):
        train_start_time = datetime.now()
        self.train_start_time = train_start_time
        if self.talk:
            print(f"Training started ({train_start_time})")

        # preparing embedding to visualise output clusters
        # using data_out because it's the same as data_in but in the normal 2d format
        if self.embedding == "pca":
            if self.talk:
                print(f"\tgenerating PCA embedding started ({datetime.now() - self.train_start_time})")
            pca_2d = PCA(n_components=2)
            pca_2d.fit(data_out)
            current_embedding = pca_2d.transform(data_out)
            if self.talk:
                print(f"\tgenerating PCA embedding finished ({datetime.now() - self.train_start_time})")

        elif self.embedding == "tsne":
            if self.talk:
                print(f"\tgenerating TSNE embedding started ({datetime.now() - self.train_start_time})")
            tsne_2d = TSNE()
            current_embedding = tsne_2d.fit_transform(data_out)
            if self.talk:
                print(f"\tgenerating TSNE embedding finished ({datetime.now() - self.train_start_time})")

        else:
            print("INVALID ARGUMENT GIVEN FOR EMBEDDING (should be \"pca\" or \"tsne\")")

        # prepare figure for cluster plots
        cluster_vis_rows = int(np.ceil(np.floor(self.epochs/10) / self.cluster_vis_cols))
        if self.visualise_clusters:
            fig, axs = plt.subplots(nrows=cluster_vis_rows, ncols=self.cluster_vis_cols, figsize=(10,4))
            fig.suptitle(f"Clusters on {self.embedding}")

        for i in range(self.epochs):
            train_loss, cluster_preds = self._train_loop(model, data_in, data_out)

            # generate training loss [?] used by M Huang et al by dividing by 400 instead of 317 (hard coded for this specific case)
            m_huang_loss = (train_loss * 317) / 400
            current_epoch_time = datetime.now() - train_start_time
            print(f"\nEpoch {i} ({current_epoch_time})")
            print(f"Training loss = {train_loss}")
            # will be 0 as long as all points are assigned to the same cluster
            print(f"Sum of clusters: {sum(cluster_preds)}")

            # adding 1 to get 10th epoch (9) instead of 1st epoch (0) etc
            epoch1 = i+1
            if self.visualise_clusters and epoch1%10 == 0:
                if ((self.epochs/10) / self.cluster_vis_cols) > 1:
                    ax = axs[int(np.floor(((epoch1/10)-1)/self.cluster_vis_cols))][int(((epoch1/10)-1)%self.cluster_vis_cols)]
                else:
                    ax = axs[int(((epoch1/10)-1)%self.cluster_vis_cols)]
                ax.scatter(current_embedding[:,0], current_embedding[:,1], c=cluster_preds, s=5, alpha=0.5)
                ax.set_title(f"Epoch {i}\nTraining loss = {train_loss}\nM. Huang training loss = {m_huang_loss}")
                ax.set_xlabel(f"Time: {current_epoch_time}")
        clusters_file = pd.DataFrame(torch.cat([self.batched_index[:,None], cluster_preds[:,None]], dim=1), columns=["batched_index", "predicted_cluster"])
        clusters_file.to_csv(self.clusters_file_name, index=False)
        # parameters count from huang et al 2021
        plt.suptitle(f"embedding = {self.embedding}, lr = {self.lr}, no. parameters = {sum([p.nelement() for p in model.parameters()])}")
        plt.tight_layout()
        plt.show()


        


            

        
        