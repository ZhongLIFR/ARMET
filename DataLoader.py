#!/usr/bin/env python3
# -*- coding: utf-8 -*-

my_n_neighbours = 5

# =============================================================================
# Step 0: define a function to read a graph from a single txt file
# =============================================================================

import os
import os.path as osp
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.io import read_txt_array
from torch_geometric.utils import coalesce, remove_self_loops


##define a function to read file
def read_file(folder, prefix, name, dtype=None):
    path = osp.join(folder, f'{prefix}_{name}.txt')
    print(path)
    return read_txt_array(path, sep=',', dtype=dtype)


##define a function to combine items into sequences
def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [item for item in seq if item.numel() > 0]
    seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
    return torch.cat(seq, dim=-1) if len(seq) > 0 else None

##define a funtion to split data into batches
def split(data, batch):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    print("----row----")
    print(row)
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    
    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)

    ##define a slices
    slices = {'edge_index': edge_slice}
        
    if data.x is not None:
        slices['x'] = node_slice
    else:
        # Imitate `collate` functionality:
        data._num_nodes = torch.bincount(batch).tolist()
        data.num_nodes = batch.numel()
                    
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
        
    return data, slices

##IMPORTANT function 1: define a function to read data from text files
def read_tu_data(folder, prefix):
    
    # =============================================================================
    # read edge index from adj matrix
    # =============================================================================
    edge_index = read_file(folder, prefix, 'A', torch.long).t() - 1 
    
    # =============================================================================
    # read graph index
    # =============================================================================    
    batch = read_file(folder, prefix, 'graph_indicator', torch.long) - 1

    # =============================================================================
    # read node attributes
    # =============================================================================    
    node_attributes = torch.empty((batch.size(0), 0))
    node_attributes = read_file(folder, prefix, 'node_attributes')

    # =============================================================================
    # read graph labels
    # =============================================================================
    y = read_file(folder, prefix, 'graph_labels', torch.long)

                
    data = Data(x=node_attributes, edge_index=edge_index , y=y)
        
    data, slices = split(data, batch)

    sizes = {'num_node_attributes': node_attributes.size(-1)}
   
    return data, slices, sizes


# =============================================================================
# Step 1: define a class to read all text file based on read_tu_data() function
# =============================================================================
from typing import Callable, List, Optional
from torch_geometric.data import InMemoryDataset


class ParseDataset(InMemoryDataset):
    
    def __init__(self,
                 root: str,
                 name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 cleaned: bool = False):
        
        self.name = name
        self.cleaned = cleaned
        super().__init__(root, transform, pre_transform, pre_filter)

        load_data = torch.load(self.processed_paths[0])
              
        self.data, self.slices, self.sizes = load_data
        
        num_node_attributes = self.num_node_attributes
        self.data.x = self.data.x[:, :num_node_attributes]
                

    @property
    def raw_dir(self) -> str:
        name = f'raw{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        name = f'processed{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, self.name, name)

    @property
    def num_node_attributes(self) -> int:
        return self.sizes['num_node_attributes']


    @property
    def raw_file_names(self) -> List[str]:
        names = ['A', 'graph_indicator']
        return [f'{self.name}_{name}.txt' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'
    
    ##renove ~/processed/ directory to run this 
    def process(self):
        self.data, self.slices, sizes = read_tu_data(self.raw_dir, self.name)
 
        if self.pre_filter is not None or self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            self.data, self.slices = self.collate(data_list)
            self._data_list = None  # Reset cache.

        torch.save((self.data, self.slices, sizes), self.processed_paths[0])
        
    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'
   

# =============================================================================
# Step 2: define a function to create data loader based on ParseDataset class
# =============================================================================

from torch_geometric.data import DataLoader, DenseDataLoader

DATA_PATH = 'Data'

if not os.path.isdir(DATA_PATH):
    os.mkdir(DATA_PATH)
else:
    print(DATA_PATH)


##IMPORTANT function: define a function to load dataset
def load_source_data(data_name):
    """
    
    Modify it to load both source and target datasets
    
    """
        
    ##get raw dataset if it already exists
    print(DATA_PATH + "/" + data_name + "/raw/")
    if os.path.exists(DATA_PATH + "/" + data_name + "/raw/"):
        
        print("++++++++find dataset++++++++++++")
        
        dataset_raw = ParseDataset(root=DATA_PATH, name=data_name)

    dataset = dataset_raw
    
    dataset_list = [data for data in dataset]
    
    normal_indices = [i for i, data in enumerate(dataset_list) if  data.y.item()==0 ]
    dataset_list = [dataset_list[idx] for idx in normal_indices]
    
    return dataset_list


##IMPORTANT function: define a function to load dataset
def load_target_data(data_name, is_val = False):
    """
    
    Modify it to load both source and target datasets
    
    """
        
    ##get raw dataset if it already exists
    print(DATA_PATH + "/" + data_name + "/raw/")
    if os.path.exists(DATA_PATH + "/" + data_name + "/raw/"):
        
        print("++++++++find dataset++++++++++++")
        
        dataset_raw = ParseDataset(root=DATA_PATH, name=data_name)

    dataset = dataset_raw
    
    dataset_list = [data for data in dataset]
    
    return dataset_list


##IMPORTANT function: define a function to load dataset
def load_val_test_data(data_name, is_val = False):
    """
    
    Modify it to load both source and target datasets
    
    """
        
    ##get raw dataset if it already exists
    print(DATA_PATH + "/" + data_name + "/raw/")
    if os.path.exists(DATA_PATH + "/" + data_name + "/raw/"):
        
        print("++++++++find dataset++++++++++++")
        
        dataset_raw = ParseDataset(root=DATA_PATH, name=data_name)

    dataset = dataset_raw
    
    dataset_list = [data for data in dataset]
    
    
    seed= 1213
    np.random.seed(seed)
    newcoin = np.random.default_rng(seed)
    torch.manual_seed(seed)
    val_ratio = 0.2
    
    normal_indices = [i for i, data in enumerate(dataset_list) if  data.y.item()==0]
    
    abnormal_indices = [i for i, data in enumerate(dataset_list) if  data.y.item()==1]
    
    all_indices = normal_indices + abnormal_indices
    
    normal_val_indices = [i for i, data in enumerate(dataset_list) if  data.y.item()==0 and newcoin.random()<val_ratio]
    
    abnormal_val_indices = [i for i, data in enumerate(dataset_list) if  data.y.item()==1 and newcoin.random()<val_ratio]
    
    all_val_indices = normal_val_indices + abnormal_val_indices 
    
    all_test_indices = [idx for idx in all_indices if idx not in all_val_indices]
    
    
    if is_val == True: ##return validation dataset
        dataset_list = [dataset_list[idx] for idx in all_val_indices]
        
    else: ##return test dataset
        dataset_list = [dataset_list[idx] for idx in all_test_indices]
    
    return dataset_list




##define a function as dataloader
def create_loaders(source_data_name, 
                   target_data_name,
                   test_data_name,
                   batch_size=64,
                   is_val = False):

    ##load source and target dataset respectively
    source_dataset = load_source_data(source_data_name)
    target_dataset = load_target_data(target_data_name) ##it is always False in validation or test
    test_dataset = load_val_test_data(target_data_name, is_val = is_val)
    
    print("Distribution of classes:")
    
    labels = np.array([data.y.item() for data in source_dataset])
    label_dist = ['%d'% (labels==c).sum() for c in [0,1]]
    print("Source: Number of graphs: %d, Class distribution %s"%(len(source_dataset), label_dist))
    
    labels = np.array([data.y.item() for data in target_dataset])
    label_dist = ['%d'% (labels==c).sum() for c in [0,1]]
    print("Target: Number of graphs: %d, Class distribution %s"%(len(target_dataset), label_dist))
    
    labels = np.array([data.y.item() for data in test_dataset])
    label_dist = ['%d'% (labels==c).sum() for c in [0,1]]
    print("Test/Val: Number of graphs: %d, Class distribution %s"%(len(test_dataset), label_dist))

    Loader =  DataLoader
    
    num_workers = 0
    
    ##----create a batch-based source dataset loader----##
    source_loader = Loader(source_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    
    ##----create a batch-based target dataset loader----##
    target_loader = Loader(target_dataset, batch_size=batch_size, shuffle=True,  pin_memory=True, num_workers=num_workers)
            
    ##----create a batch-based target dataset loader----##
    ##importantly, the test_loader should not be shuffled!
    test_loader = Loader(test_dataset, batch_size=batch_size, shuffle=False,  pin_memory=True, num_workers=num_workers)

    return source_loader, target_loader, test_loader, target_dataset[0].num_features, source_dataset, target_dataset, test_dataset


# =============================================================================
# Step 3: create a trainer class to train NN, including train() and test()
# =============================================================================
from sklearn.metrics import average_precision_score, roc_auc_score

class HPSelection:
    
    # =============================================================================
    # Step1. initialise the trainer with given hyperparameters
    # =============================================================================
    def __init__(self, 
                 source_data_name,
                 target_data_name,
                 se_model, 
                 st_model, 
                 source_classifier,
                 domain_classifier,
                 optimizer_se, 
                 optimizer_st, 
                 optimizer_dc,
                 sc_weight,
                 dc_weight,
                 ca_weight,
                 device=torch.device("cpu"), 
                 regularizer="variance"):
        
        self.device = device
        
        ##Data names
        self.source_data_name = source_data_name
        self.target_data_name = target_data_name
        
        ##Feature Extractor
        self.se_model = se_model
        self.st_model = st_model
        self.optimizer_se = optimizer_se
        self.optimizer_st = optimizer_st


        ##Source Classifier
        self.source_classifier = source_classifier
        self.sc_weight = sc_weight
        
        ##Domain Classifier
        self.domain_classifier = domain_classifier
        self.optimizer_dc = optimizer_dc    
        self.dc_weight = dc_weight
        
        ##Classifier Aligner
        self.ca_weight = ca_weight
        
        ##now we have two models composed in the form f(g(input)), how to perform bp?

        ##--parameters for OCSVDD objectives----##
        self.center = None
        self.regularizer = regularizer   
    
    # =============================================================================
    # Step2. define the train funtion which will use both 
    # =============================================================================
    def train(self, 
              source_data_name,
              target_data_name,
              source_loader, 
              target_loader, 
              sc_weight,
              dc_weight,
              ca_weight,
              TSNE_plot):
        
        self.se_model.train()
        self.st_model.train()
        self.source_classifier.train()
        self.domain_classifier.train()
        
        
        ##----first iteration, define s list to store vectors for computing SVDD center---##
        if self.center == None:
            F_list = []

        total_loss_accum = 0
        total_iters = 0
        
        # =============================================================================
        # =============================================================================
                
        batch_num = 0 ##to denote the batch number
        
        # ====================================
        ## Trace loss to set weights start
        # ====================================
        sc_loss_accum = 0
        dc_loss_accum = 0
        ca_loss_accum = 0
        # ====================================
        ## Trace loss to set weights end
        # ====================================
        
        from itertools import zip_longest
        # for source_batch, target_batch in zip_longest(source_loader, target_loader, fillvalue=None):
        for source_batch, target_batch in zip(source_loader, target_loader):
            
            batch_num += 1
    
            
            # =============================================================================
            #  MODULE 1: Feature Extraction          
            # =============================================================================
            # -----------------------------------------------------------------
            # Step1. use a GIN + readout to learn SEMANTIC FEATURES for source domain (ThetaSE)          
            # -----------------------------------------------------------------
            ##----use GIN model to obatin node embeddings----##
            source_embeddings = self.se_model(source_batch)
            
            ##----use mean Readout to obtain graph embeddings----##
            mean_source_embeddings = [torch.mean(emb, dim=0) for emb in source_embeddings]
                        
            # -----------------------------------------------------------------
            # Step2. use a paramters-shared  GIN + readout to learn SEMANTIC FEATURES for target domain (ThetaSE)           
            # -----------------------------------------------------------------
            ##----use GIN model to obatin node embeddings----##
            target_embeddings = self.se_model(target_batch)
            
            ##----use mean Readout to obtain graph embeddings----##
            mean_target_embeddings = [torch.mean(emb, dim=0) for emb in target_embeddings]         
            
            
            # -----------------------------------------------------------------
            # Step3. construct KNN graph for source domain          
            # -----------------------------------------------------------------
            ##----convert list of tensors to dataframe----##
            mean_source_embeddings_list = [a.tolist() for a in mean_source_embeddings]
            import pandas as  pd
            df_mean_source_embeddings = pd.DataFrame.from_records(mean_source_embeddings_list)
            
            ##----generate KNN graph from a dataframe----##
            ##Nodes - Individual Graphs (by Index)
            ##Edges - An Edge Between Graph A and Graph B if either A in KNN(B) or B in KNN(A)
            ##Node Attributes - The embeddings of A Graph
            
            import numpy as np
            import pandas as pd
            from sklearn.neighbors import KDTree ## https://scikit-learn.org/stable/modules/neighbors.html
            from torch_geometric.data import Data
            from torch_geometric.data import DataLoader
            
            ##define attributes for all node
            df_knn_attributes_source = df_mean_source_embeddings
            df_knn_attributes_list_source = df_knn_attributes_source.values.tolist()
            
            ##define edges 
            kdt_source = KDTree(df_knn_attributes_source, leaf_size=30, metric='euclidean')
            n_neighbours = my_n_neighbours ##for Group1
            # n_neighbours = 1
            df_knn_edges_source = kdt_source.query(df_knn_attributes_source, k=n_neighbours, return_distance=False)
            df_knn_edges_list_source = []
            for i in range(0, len(df_knn_edges_source)):
                for j in range(0, n_neighbours):
                    df_knn_edges_list_source.append([i,df_knn_edges_source[i][j]])
            df_knn_edges_list_source = [list(i) for i in zip(*df_knn_edges_list_source)] ##transpose it
            
            ##define KNN graph 
            source_x = torch.tensor(df_knn_attributes_list_source, dtype=torch.float)
            source_edge_index = torch.tensor(df_knn_edges_list_source, dtype=torch.long)
            source_st_data = Data(x=source_x, edge_index=source_edge_index)
            
            # -----------------------------------------------------------------
            # Step4. use GIN2 to learn STRUCTURE FEATURES for source domain (ThetaST)          
            # -----------------------------------------------------------------
            source_st_loader = DataLoader([source_st_data], batch_size=1, shuffle=True, pin_memory=True, num_workers=0)
            
            for my_batch in source_st_loader:
                source_st_embeddings = self.st_model(my_batch)
        
            source_st_embeddings = source_st_embeddings[0]
            
            # -----------------------------------------------------------------
            # Step5. construct KNN graph for target domain          
            # -----------------------------------------------------------------
            ##----convert list of tensors to dataframe----##
            mean_target_embeddings_list = [a.tolist() for a in mean_target_embeddings]
            df_mean_target_embeddings = pd.DataFrame.from_records(mean_target_embeddings_list)
            
            
            ##define attributes for all node
            df_knn_attributes_target = df_mean_target_embeddings
            df_knn_attributes_list_target = df_knn_attributes_target.values.tolist()
            
            ##define edges 
            kdt_target = KDTree(df_knn_attributes_target, leaf_size=30, metric='euclidean')
            n_neighbours = my_n_neighbours
            df_knn_edges_target = kdt_target.query(df_knn_attributes_list_target, k=n_neighbours, return_distance=False)
            df_knn_edges_list_target = []
            for i in range(0, len(df_knn_edges_target)):
                for j in range(0, n_neighbours):
                    df_knn_edges_list_target.append([i,df_knn_edges_target[i][j]])
            df_knn_edges_list_target = [list(i) for i in zip(*df_knn_edges_list_target)] ##transpose it
            
            ##define KNN graph 
            target_x = torch.tensor(df_knn_attributes_list_target, dtype=torch.float)
            target_edge_index = torch.tensor(df_knn_edges_list_target, dtype=torch.long)
            target_st_data = Data(x=target_x, edge_index=target_edge_index)

    
            # -----------------------------------------------------------------
            # Step6. use GIN2 to learn STRUCTURE FEATURES for target domain (ThetaST)          
            # -----------------------------------------------------------------
            target_st_loader = DataLoader([target_st_data], batch_size=1, shuffle=True, pin_memory=True, num_workers=0)
            
            for my_batch in target_st_loader:
                target_st_embeddings = self.st_model(my_batch)
        
            target_st_embeddings = target_st_embeddings[0]

            # -----------------------------------------------------------------
            # Step7. concatenate the SEMANTIC FEATURES and STRUCTURE FEATURES for source domain          
            # -----------------------------------------------------------------
            
            source_concat_tensor = [torch.cat([a,b], dim=0) for a, b in zip(mean_source_embeddings, source_st_embeddings)]
            
            # -----------------------------------------------------------------
            # Step8. concatenate the SEMANTIC FEATURES and STRUCTURE FEATURES for target domain          
            # -----------------------------------------------------------------
            
            target_concat_tensor = [torch.cat([a,b], dim=0) for a, b in zip(mean_target_embeddings, target_st_embeddings)]
            
            # =============================================================================
            #  MODULE 2: Cross Domain Graph-Level Anomaly Detection      
            # =============================================================================
                        
            source_train = torch.stack(source_concat_tensor)
            
            sc_true = source_batch.y
            sc_true = sc_true.unsqueeze(1)
            
            ##----if first iteration, store vectors for computing SVDD center, and do not perform any backprop----##
            if self.center == None:
                F_list.append(source_train)
            
            ##----if not first iteration, perform backprop----##
            else:
                
                # -----------------------------------------------------------------
                # Step1. source classifier using SVDD, no update 
                # -----------------------------------------------------------------
               
                source_train_scores = torch.sum((source_train - self.center)**2, dim=1).cpu() 
                
                ##the second term in SVDD objective is controled by regularizer automatically
                sc_loss = torch.mean(source_train_scores)
          
                sc_loss_accum += sc_loss.detach().cpu().numpy()
                     
                # -----------------------------------------------------------------
                # Step2. domain classifier using MLP+Sigmoid, update ThetaDC
                # -----------------------------------------------------------------  
                ##the update of this step may repeat Nd times in each batch
                
                source_concat_tensor_clone2 = [tensor.clone() for tensor in source_concat_tensor]
    
                target_concat_tensor_clone = [tensor.clone() for tensor in target_concat_tensor]
    
                domain_label = len(source_concat_tensor_clone2)*[0]+len(target_concat_tensor_clone)*[1]
                
                source_target_con_clone = source_concat_tensor_clone2 + target_concat_tensor_clone
                
                dc_results = []
                ##perform prediction and append the result
                for graph_id in range(0,len(source_target_con_clone)):
                    temp_res = self.domain_classifier(source_target_con_clone[graph_id])
                    dc_results.append(temp_res)
                                                            
                dc_predicted = torch.stack(dc_results)
                dc_true = torch.tensor(domain_label)
                dc_true = dc_true.unsqueeze(1)
                
                dc_loss_function = nn.BCELoss()
                dc_loss  = dc_loss_function(dc_predicted, dc_true.float())
                
                ##Backpropagate for ThetaDC               
                dc_loss_accum += dc_loss.detach().cpu().numpy()
                
                
                self.optimizer_dc.zero_grad()
                dc_loss.backward(retain_graph=True)         
                self.optimizer_dc.step()
                
                
                # -----------------------------------------------------------------
                # Step3. using distance to the center to obtain psudolabels, no update    
                # ----------------------------------------------------------------- 
                target_train = torch.stack(target_concat_tensor)             
                target_train_scores = torch.sum((target_train - self.center)**2, dim=1).cpu()    
                                                                
                
                # Use q95 as the threshold for defining pseudolabels 
                label_threhold =  np.quantile(target_train_scores.detach().numpy(), [0.95])[0]


                target_pseudo_label = (target_train_scores > label_threhold).float()
                
         
                # -----------------------------------------------------------------
                # Step4. class aligner, no  update 
                # ----------------------------------------------------------------- 
                ##我们可以添加SSL的计算PsudoLabel的做法
                source_true_label = sc_true
                
                source_embeddings = torch.stack(source_concat_tensor)
                target_embeddings = torch.stack(target_concat_tensor)
                
                import torch.nn.functional as F
                
                # Calculate the mean embeddings for each label for both domains
                #------------------------------------------------------------------
                source_mean_0 = source_embeddings[source_true_label[:, 0] == 0].mean(dim=0)         
                target_mean_0 = target_embeddings[target_pseudo_label == 0].mean(dim=0)
                target_mean_1 = target_embeddings[target_pseudo_label == 1].mean(dim=0) 
    
                # using TSNE
                #------------------------------------------------------------------
                if TSNE_plot == True and batch_num == 1:
                                    
                    X1 = source_embeddings[source_true_label[:, 0] == 0].detach().numpy()
                    X2 = target_embeddings[target_pseudo_label == 0].detach().numpy()
                    X3 = target_embeddings[target_pseudo_label == 1].detach().numpy()
        
                    from sklearn.manifold import TSNE
                    import matplotlib.pyplot as plt
                                    
                    # combine the tensors into a single matrix
                    X = np.vstack((X1, X2, X3))
                    
                    # perform t-SNE on the combined matrix
                    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
                    tsne_results = tsne.fit_transform(X)
                    
                    # create a scatter plot of the t-SNE results, with different colors for each tensor
                    plt.scatter(tsne_results[:len(X1), 0], tsne_results[:len(X1), 1], c='blue', label='source_normal', marker = "o", alpha=0.5)
                    plt.scatter(tsne_results[len(X1):len(X1)+len(X2), 0], tsne_results[len(X1):len(X1)+len(X2), 1], c='green', label='target_normal', marker = "o", alpha=0.5)
                    plt.scatter(tsne_results[len(X1)+len(X2):len(X1)+len(X2)+len(X3), 0], tsne_results[len(X1)+len(X2):len(X1)+len(X2)+len(X3), 1], c='red', label='target_abnormal', marker = "X", alpha=0.5)
                    plt.legend(ncol=1, bbox_to_anchor=(-0.15, 1.0), loc='upper left')
                    plt.axis('off')
                    plt.show()
                                
                    
                ##Error2: all target have "0" as pseudolabels, target_mean_1 will be "nan"
                ##-------------------------------------
                ## Compute the Euclidean distance between the mean embeddings
                inter_dist_0 = F.pairwise_distance(source_mean_0.unsqueeze(0), target_mean_0.unsqueeze(0))
                
                intra_dist_t = F.pairwise_distance(target_mean_0.unsqueeze(0), target_mean_1.unsqueeze(0))
                
                intra_dist_t = intra_dist_t if not torch.isnan(intra_dist_t) else torch.tensor([0]) ##set to zero it is nan
                
                ca_loss = inter_dist_0 - intra_dist_t 
    
                ca_loss_accum += ca_loss.detach().cpu().numpy()
                
                
                #=============================================================================
                # Step5. compute total loss and update parameters ThetaSE, ThetaST
                # =============================================================================
                
                #--------------------------------------------------
                ##compute dc_loss using updated ThetaDC
                #--------------------------------------------------
    
                domain_label = len(source_concat_tensor)*[0]+len(target_concat_tensor)*[1]          
                source_target_con = source_concat_tensor + target_concat_tensor
                
                dc_results = []
                for graph_id in range(0,len(source_target_con)):
                    temp_res = self.domain_classifier(source_target_con[graph_id])
                    dc_results.append(temp_res)
                                                            
                dc_predicted = torch.stack(dc_results)
                dc_true = torch.tensor(domain_label)
                dc_true = dc_true.unsqueeze(1)
                
                dc_loss_function = nn.BCELoss()
                dc_loss  = dc_loss_function(dc_predicted, dc_true.float()) 
                
                #--------------------------------------------------
                ##combine sc_loss, dc_loss, ca_loss to update ThetaST and ThetaSE
                #--------------------------------------------------
                
                ##Setting weights of different lamdas for different transfer tasks in a heuristic manner                
                sc_weight = sc_weight
                dc_weight = dc_weight
                ca_weight = ca_weight
                 

                ##with tc_loss:
                total_loss = sc_weight*sc_loss - dc_weight*dc_loss + ca_weight*ca_loss
                
                                    
                ###Backpropagate for ThetaSE and ThetaST 
                self.optimizer_se.zero_grad()
                self.optimizer_st.zero_grad()
                # self.optimizer_all.zero_grad()
                total_loss.backward()    
                # self.optimizer_all.step()
                self.optimizer_st.step()
                self.optimizer_se.step()       
                
                total_loss_accum += total_loss.detach().cpu().numpy()
                total_iters += 1
                    
        ##----first epoch only, compute SVDD center----##
        if self.center == None: ##first epoch only
            full_F_list = torch.cat(F_list)
            self.center = torch.mean(full_F_list, dim=0).detach() #no backpropagation for center
            
            average_sc_loss = -1
            average_dc_loss = -1
            average_ca_loss = -1
            average_total_loss = -1
        ##----if not first epoch, compute averaged SVDD loss ----##
        else:
            average_sc_loss = sc_loss_accum/total_iters
            average_dc_loss = dc_loss_accum/total_iters
            average_ca_loss = ca_loss_accum/total_iters
            average_total_loss = total_loss_accum/total_iters

        return average_sc_loss,average_dc_loss,average_ca_loss,average_total_loss
    

    
    def test(self, test_loader):
        self.se_model.eval()
        
        with torch.no_grad():

            prediction_list = []
            for batch in test_loader:
                
                # -----------------------------------------------------------------
                # Step1. use a GIN + readout to learn SEMANTIC FEATURES for source domain (ThetaSE)          
                # -----------------------------------------------------------------
                test_embeddings = self.se_model(batch)                
                mean_test_embeddings = [torch.mean(emb, dim=0) for emb in test_embeddings]


                # -----------------------------------------------------------------
                # Step2. construct KNN graph for target domain          
                # -----------------------------------------------------------------
                import pandas as pd
                from sklearn.neighbors import KDTree 
                mean_test_embeddings_list = [a.tolist() for a in mean_test_embeddings]
                df_mean_test_embeddings = pd.DataFrame.from_records(mean_test_embeddings_list)
                
                ##define attributes for all node
                df_knn_attributes_test = df_mean_test_embeddings
                df_knn_attributes_list_test = df_knn_attributes_test.values.tolist()
                
                ##define edges 
                kdt_test = KDTree(df_knn_attributes_test, leaf_size=30, metric='euclidean')
                n_neighbours = 5
                df_knn_edges_test = kdt_test.query(df_knn_attributes_list_test, k=n_neighbours, return_distance=False)
                df_knn_edges_list_test = []
                for i in range(0, len(df_knn_edges_test)):
                    for j in range(0, n_neighbours):
                        df_knn_edges_list_test.append([i,df_knn_edges_test[i][j]])
                df_knn_edges_list_test = [list(i) for i in zip(*df_knn_edges_list_test)] ##transpose it
                
                ##define KNN graph 
                test_x = torch.tensor(df_knn_attributes_list_test, dtype=torch.float)
                test_edge_index = torch.tensor(df_knn_edges_list_test, dtype=torch.long)
                test_st_data = Data(x=test_x, edge_index=test_edge_index)
                
                
                # -----------------------------------------------------------------
                # Step3. use GIN2 to learn STRUCTURE FEATURES for test domain (ThetaST)          
                # -----------------------------------------------------------------
                test_st_loader = DataLoader([test_st_data], batch_size=1, shuffle=True, pin_memory=True, num_workers=0)
                
                for my_batch in test_st_loader:
                    test_st_embeddings = self.st_model(my_batch)
            
                test_st_embeddings = test_st_embeddings[0]

                # -----------------------------------------------------------------
                # Step4. concatenate the SEMANTIC FEATURES and STRUCTURE FEATURES for test domain          
                # -----------------------------------------------------------------
                
                test_concat_tensor = [torch.cat([a,b], dim=0) for a, b in zip(mean_test_embeddings, test_st_embeddings)]
    
                target_test = torch.stack(test_concat_tensor) 
                # -----------------------------------------------------------------
                # Step5. using SVDD center to predict labels and save prediction results
                # ----------------------------------------------------------------- 
                test_predict_results = torch.sum((target_test - self.center)**2, dim=1).cpu()
                                                                                         
                prediction_list.append(test_predict_results)

           
            labels = torch.cat([batch.y for batch in test_loader])
            preds = torch.cat(prediction_list)
            # print(labels)
            # print(preds)

            ap = average_precision_score(y_true= labels, y_score= preds, average = None, pos_label= 1, sample_weight= None)
            roc_auc = roc_auc_score(y_true= labels, y_score= preds, average = None,
                                    sample_weight= None, max_fpr = None, 
                                    multi_class = 'raise', labels =None)

            return ap, roc_auc, preds, labels
        
        
    
class MyTrainer:
    
    # =============================================================================
    # Step1. initialise the trainer with given hyperparameters
    # =============================================================================
    def __init__(self, 
                 source_data_name,
                 target_data_name,
                 se_model, 
                 st_model, 
                 source_classifier,
                 domain_classifier,
                 optimizer_se, 
                 optimizer_st, 
                 optimizer_dc,
                 sc_weight,
                 dc_weight,
                 ca_weight,
                 device=torch.device("cpu"), 
                 regularizer="variance"):
        
        self.device = device
        
        ##Data names
        self.source_data_name = source_data_name
        self.target_data_name = target_data_name
        
        ##Feature Extractor
        self.se_model = se_model
        self.st_model = st_model
        self.optimizer_se = optimizer_se
        self.optimizer_st = optimizer_st


        ##Source Classifier
        self.source_classifier = source_classifier
        self.sc_weight = sc_weight
        
        ##Domain Classifier
        self.domain_classifier = domain_classifier
        self.optimizer_dc = optimizer_dc    
        self.dc_weight = dc_weight
        
        ##Classifier Aligner
        self.ca_weight = ca_weight
        
        ##now we have two models composed in the form f(g(input)), how to perform bp?

        ##--parameters for OCSVDD objectives----##
        self.center = None
        self.regularizer = regularizer   
    
    # =============================================================================
    # Step2. define the train funtion which will use both 
    # =============================================================================
    def train(self, 
              source_data_name,
              target_data_name,
              source_loader, 
              target_loader, 
              sc_weight,
              dc_weight,
              ca_weight,
              TSNE_plot):
        """
        =============================================================================
        MODULE 1: Feature Extraction
        =============================================================================
        for each batch (batch size should be large enough to include both positive and negative samples), we do:
        1. use a GIN + readout to learn SEMANTIC FEATURES for source domain (ThetaSE)
        2. use a paramters-shared  GIN + readout to learn SEMANTIC FEATURES for target domain (ThetaSE)
        3. construct KNN graph for source domain
        4. construct KNN graph for target domain
        5. use GIN2 to learn STRUCTURE FEATURES for source domain (ThetaST)
        6. use a paramters-shared  GIN2 to learn STRUCTURE FEATURES for target domain (ThetaST)
        7. concatenate the SEMANTIC FEATURES and STRUCTURE FEATURES for source domain
        8. concatenate the SEMANTIC FEATURES and STRUCTURE FEATURES for target domain
        
        ThetaFE = (ThetaSE;ThetaST)
        
        =============================================================================
        MODULE 2: Cross Domain Graph-Level Anomaly Detection
        =============================================================================
        1. constuct a source classifier for source domain (ThetaSC)[LossSC]
        2. construct a domain classifier for source and target domains (ThetaDC)[LossDC]
        3. generate pseudolabels for target domain 
        4. construct class aligner [LossCA]
        5. compute TotalLoss = weight1*LossSC -weight2*LossSC + weight3*LossCA [TotalLoss]
        
        =============================================================================
        UPDATE: (This needs to be carefully designed)
        =============================================================================
        1. Initialise all parameters
        2. Backpropagation LossSC and update ThetaSC
        3. Backpropagation LossDC and update ThetaDC
        4. Backpropagation TotalLoss and update ThetaFE
        
        """
        print("\n++++++++++++++++trainers.py++++++++++++++++")
        print("----------train()----------")
        self.se_model.train()
        self.st_model.train()
        self.source_classifier.train()
        self.domain_classifier.train()
        
        ##----first iteration, define s list to store vectors for computing SVDD center---##
        if self.center == None:
            F_list = []

        total_loss_accum = 0
        total_iters = 0
        
        # =============================================================================
        # =============================================================================
        print(len(source_loader))
        print(len(target_loader))
        
        
        batch_num = 0 ##to denote the batch number
        
        # ====================================
        ## Trace loss to set weights start
        # ====================================
        sc_loss_accum = 0
        dc_loss_accum = 0
        ca_loss_accum = 0
        # ====================================
        ## Trace loss to set weights end
        # ====================================
        
        from itertools import zip_longest
        # for source_batch, target_batch in zip_longest(source_loader, target_loader, fillvalue=None):
        for source_batch, target_batch in zip(source_loader, target_loader):
            
            batch_num += 1
    
            print("\n++++++++++++++++trainers.py++++++++++++++++")
            print("----------batch {} training start----------".format(batch_num))
            print(source_batch)
            # print(source_batch[0])
            
            # =============================================================================
            #  MODULE 1: Feature Extraction          
            # =============================================================================
            # -----------------------------------------------------------------
            # Step1. use a GIN + readout to learn SEMANTIC FEATURES for source domain (ThetaSE)          
            # -----------------------------------------------------------------
            ##----use GIN model to obatin node embeddings----##
            source_embeddings = self.se_model(source_batch)
            
            ##----use mean Readout to obtain graph embeddings----##
            mean_source_embeddings = [torch.mean(emb, dim=0) for emb in source_embeddings]
                        
            # -----------------------------------------------------------------
            # Step2. use a paramters-shared  GIN + readout to learn SEMANTIC FEATURES for target domain (ThetaSE)           
            # -----------------------------------------------------------------
            ##----use GIN model to obatin node embeddings----##
            target_embeddings = self.se_model(target_batch)
            
            ##----use mean Readout to obtain graph embeddings----##
            mean_target_embeddings = [torch.mean(emb, dim=0) for emb in target_embeddings]         
            
            
            # -----------------------------------------------------------------
            # Step3. construct KNN graph for source domain          
            # -----------------------------------------------------------------
            ##----convert list of tensors to dataframe----##
            mean_source_embeddings_list = [a.tolist() for a in mean_source_embeddings]
            import pandas as  pd
            df_mean_source_embeddings = pd.DataFrame.from_records(mean_source_embeddings_list)
            
            ##----generate KNN graph from a dataframe----##
            ##Nodes - Individual Graphs (by Index)
            ##Edges - An Edge Between Graph A and Graph B if either A in KNN(B) or B in KNN(A)
            ##Node Attributes - The embeddings of A Graph
            
            import numpy as np
            import pandas as pd
            from sklearn.neighbors import KDTree ## https://scikit-learn.org/stable/modules/neighbors.html
            from torch_geometric.data import Data
            from torch_geometric.data import DataLoader
            
            ##define attributes for all node
            df_knn_attributes_source = df_mean_source_embeddings
            df_knn_attributes_list_source = df_knn_attributes_source.values.tolist()
            
            ##define edges 
            kdt_source = KDTree(df_knn_attributes_source, leaf_size=30, metric='euclidean')
            n_neighbours = my_n_neighbours ##for Group1
            # n_neighbours = 1
            df_knn_edges_source = kdt_source.query(df_knn_attributes_source, k=n_neighbours, return_distance=False)
            df_knn_edges_list_source = []
            for i in range(0, len(df_knn_edges_source)):
                for j in range(0, n_neighbours):
                    df_knn_edges_list_source.append([i,df_knn_edges_source[i][j]])
            df_knn_edges_list_source = [list(i) for i in zip(*df_knn_edges_list_source)] ##transpose it
            
            ##define KNN graph 
            source_x = torch.tensor(df_knn_attributes_list_source, dtype=torch.float)
            source_edge_index = torch.tensor(df_knn_edges_list_source, dtype=torch.long)
            source_st_data = Data(x=source_x, edge_index=source_edge_index)
            
            # -----------------------------------------------------------------
            # Step4. use GIN2 to learn STRUCTURE FEATURES for source domain (ThetaST)          
            # -----------------------------------------------------------------
            source_st_loader = DataLoader([source_st_data], batch_size=1, shuffle=True, pin_memory=True, num_workers=0)
            
            for my_batch in source_st_loader:
                source_st_embeddings = self.st_model(my_batch)
        
            source_st_embeddings = source_st_embeddings[0]
            print(len(source_st_embeddings)) ##only one KNN-graph
            print(source_st_embeddings[0].size())
            

            # -----------------------------------------------------------------
            # Step5. construct KNN graph for target domain          
            # -----------------------------------------------------------------
            ##----convert list of tensors to dataframe----##
            mean_target_embeddings_list = [a.tolist() for a in mean_target_embeddings]
            df_mean_target_embeddings = pd.DataFrame.from_records(mean_target_embeddings_list)
            
            
            ##define attributes for all node
            df_knn_attributes_target = df_mean_target_embeddings
            df_knn_attributes_list_target = df_knn_attributes_target.values.tolist()
            
            ##define edges 
            kdt_target = KDTree(df_knn_attributes_target, leaf_size=30, metric='euclidean')
            n_neighbours = my_n_neighbours
            df_knn_edges_target = kdt_target.query(df_knn_attributes_list_target, k=n_neighbours, return_distance=False)
            df_knn_edges_list_target = []
            for i in range(0, len(df_knn_edges_target)):
                for j in range(0, n_neighbours):
                    df_knn_edges_list_target.append([i,df_knn_edges_target[i][j]])
            df_knn_edges_list_target = [list(i) for i in zip(*df_knn_edges_list_target)] ##transpose it
            
            ##define KNN graph 
            target_x = torch.tensor(df_knn_attributes_list_target, dtype=torch.float)
            target_edge_index = torch.tensor(df_knn_edges_list_target, dtype=torch.long)
            target_st_data = Data(x=target_x, edge_index=target_edge_index)

    
            # -----------------------------------------------------------------
            # Step6. use GIN2 to learn STRUCTURE FEATURES for target domain (ThetaST)          
            # -----------------------------------------------------------------
            target_st_loader = DataLoader([target_st_data], batch_size=1, shuffle=True, pin_memory=True, num_workers=0)
            
            for my_batch in target_st_loader:
                target_st_embeddings = self.st_model(my_batch)
        
            target_st_embeddings = target_st_embeddings[0]

            # -----------------------------------------------------------------
            # Step7. concatenate the SEMANTIC FEATURES and STRUCTURE FEATURES for source domain          
            # -----------------------------------------------------------------
            
            source_concat_tensor = [torch.cat([a,b], dim=0) for a, b in zip(mean_source_embeddings, source_st_embeddings)]
            
            print(len(source_concat_tensor))
            print(source_concat_tensor[0].size())
            
            
            # -----------------------------------------------------------------
            # Step8. concatenate the SEMANTIC FEATURES and STRUCTURE FEATURES for target domain          
            # -----------------------------------------------------------------
            
            target_concat_tensor = [torch.cat([a,b], dim=0) for a, b in zip(mean_target_embeddings, target_st_embeddings)]
            
            # =============================================================================
            #  MODULE 2: Cross Domain Graph-Level Anomaly Detection      
            # =============================================================================
                        
            source_train = torch.stack(source_concat_tensor)
            
            sc_true = source_batch.y
            sc_true = sc_true.unsqueeze(1)
            
            ##----if first iteration, store vectors for computing SVDD center, and do not perform any backprop----##
            if self.center == None:
                F_list.append(source_train)
            
            ##----if not first iteration, perform backprop----##
            else:
                
                # -----------------------------------------------------------------
                # Step1. source classifier using SVDD, no update 
                # -----------------------------------------------------------------
               
                source_train_scores = torch.sum((source_train - self.center)**2, dim=1).cpu() 
                
                ##the second term in SVDD objective is controled by regularizer automatically
                sc_loss = torch.mean(source_train_scores)

                print("sc_loss:{}".format(sc_loss.detach().numpy()))            
                sc_loss_accum += sc_loss.detach().cpu().numpy()
                     
                # -----------------------------------------------------------------
                # Step2. domain classifier using MLP+Sigmoid, update ThetaDC
                # -----------------------------------------------------------------  
                ##the update of this step may repeat Nd times in each batch
                
                source_concat_tensor_clone2 = [tensor.clone() for tensor in source_concat_tensor]
    
                target_concat_tensor_clone = [tensor.clone() for tensor in target_concat_tensor]
    
                domain_label = len(source_concat_tensor_clone2)*[0]+len(target_concat_tensor_clone)*[1]
                
                source_target_con_clone = source_concat_tensor_clone2 + target_concat_tensor_clone
                
                dc_results = []
                ##perform prediction and append the result
                for graph_id in range(0,len(source_target_con_clone)):
                    temp_res = self.domain_classifier(source_target_con_clone[graph_id])
                    dc_results.append(temp_res)
                                                            
                dc_predicted = torch.stack(dc_results)
                dc_true = torch.tensor(domain_label)
                dc_true = dc_true.unsqueeze(1)
                
                dc_loss_function = nn.BCELoss()
                dc_loss  = dc_loss_function(dc_predicted, dc_true.float())
                
                ##Backpropagate for ThetaDC 
                print("dc_loss:{}".format(dc_loss.detach().numpy()))               
                dc_loss_accum += dc_loss.detach().cpu().numpy()
                
                
                self.optimizer_dc.zero_grad()
                dc_loss.backward(retain_graph=True)         
                self.optimizer_dc.step()
                print("==update of ThetaDC==")
                
                
                # -----------------------------------------------------------------
                # Step3. using distance to the center to obtain psudolabels, no update    
                # ----------------------------------------------------------------- 
                target_train = torch.stack(target_concat_tensor)             
                target_train_scores = torch.sum((target_train - self.center)**2, dim=1).cpu()    
                                                                
                
                # Use q95 as the threshold for defining pseudolabels 
                label_threhold =  np.quantile(target_train_scores.detach().numpy(), [0.95])[0]


                target_pseudo_label = (target_train_scores > label_threhold).float()
                
         
                # -----------------------------------------------------------------
                # Step4. class aligner, no  update 
                # ----------------------------------------------------------------- 
                ##我们可以添加SSL的计算PsudoLabel的做法
                source_true_label = sc_true
                
                source_embeddings = torch.stack(source_concat_tensor)
                target_embeddings = torch.stack(target_concat_tensor)
                
                import torch.nn.functional as F
                
                # Calculate the mean embeddings for each label for both domains
                #------------------------------------------------------------------
                source_mean_0 = source_embeddings[source_true_label[:, 0] == 0].mean(dim=0)         
                target_mean_0 = target_embeddings[target_pseudo_label == 0].mean(dim=0)
                target_mean_1 = target_embeddings[target_pseudo_label == 1].mean(dim=0) 
    
                # using TSNE
                #------------------------------------------------------------------
                if TSNE_plot == True and batch_num == 1:
                                    
                    X1 = source_embeddings[source_true_label[:, 0] == 0].detach().numpy()
                    X2 = target_embeddings[target_pseudo_label == 0].detach().numpy()
                    X3 = target_embeddings[target_pseudo_label == 1].detach().numpy()
        
                    from sklearn.manifold import TSNE
                    import matplotlib.pyplot as plt
                                    
                    # combine the tensors into a single matrix
                    X = np.vstack((X1, X2, X3))
                    
                    # perform t-SNE on the combined matrix
                    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
                    tsne_results = tsne.fit_transform(X)
                    
                    # create a scatter plot of the t-SNE results, with different colors for each tensor
                    plt.scatter(tsne_results[:len(X1), 0], tsne_results[:len(X1), 1], c='blue', label='source_normal', marker = "o", alpha=0.5)
                    plt.scatter(tsne_results[len(X1):len(X1)+len(X2), 0], tsne_results[len(X1):len(X1)+len(X2), 1], c='green', label='target_normal', marker = "o", alpha=0.5)
                    plt.scatter(tsne_results[len(X1)+len(X2):len(X1)+len(X2)+len(X3), 0], tsne_results[len(X1)+len(X2):len(X1)+len(X2)+len(X3), 1], c='red', label='target_abnormal', marker = "X", alpha=0.5)
                    plt.legend(ncol=1, bbox_to_anchor=(-0.15, 1.0), loc='upper left')
                    plt.axis('off')
                    plt.show()
                                
                    
                ##Error2: all target have "0" as pseudolabels, target_mean_1 will be "nan"
                ##-------------------------------------
                ## Compute the Euclidean distance between the mean embeddings
                inter_dist_0 = F.pairwise_distance(source_mean_0.unsqueeze(0), target_mean_0.unsqueeze(0))
                
                intra_dist_t = F.pairwise_distance(target_mean_0.unsqueeze(0), target_mean_1.unsqueeze(0))
                
                intra_dist_t = intra_dist_t if not torch.isnan(intra_dist_t) else torch.tensor([0]) ##set to zero it is nan
                
                ca_loss = inter_dist_0 - intra_dist_t 
    
                print("ca_loss:{}".format(ca_loss.detach().numpy()))
                ca_loss_accum += ca_loss.detach().cpu().numpy()
                
                
                print(" -inter_dist_0:{}\n -intra_dist_t:{}\n".format(inter_dist_0.detach().numpy(),
                                                                      intra_dist_t.detach().numpy()))
                
                #=============================================================================
                # Step5. compute total loss and update parameters ThetaSE, ThetaST
                # =============================================================================
                
                #--------------------------------------------------
                ##compute dc_loss using updated ThetaDC
                #--------------------------------------------------
    
                domain_label = len(source_concat_tensor)*[0]+len(target_concat_tensor)*[1]          
                source_target_con = source_concat_tensor + target_concat_tensor
                
                dc_results = []
                for graph_id in range(0,len(source_target_con)):
                    temp_res = self.domain_classifier(source_target_con[graph_id])
                    dc_results.append(temp_res)
                                                            
                dc_predicted = torch.stack(dc_results)
                dc_true = torch.tensor(domain_label)
                dc_true = dc_true.unsqueeze(1)
                
                dc_loss_function = nn.BCELoss()
                dc_loss  = dc_loss_function(dc_predicted, dc_true.float()) 
                
                #--------------------------------------------------
                ##combine sc_loss, dc_loss, ca_loss to update ThetaST and ThetaSE
                #--------------------------------------------------
                ##with tc_loss:
                total_loss = sc_weight*sc_loss - dc_weight*dc_loss + ca_weight*ca_loss
                
            
                print("total_loss:{}".format(total_loss.detach().numpy()))
                        
                ###Backpropagate for ThetaSE and ThetaST 
                self.optimizer_se.zero_grad()
                self.optimizer_st.zero_grad()
                # self.optimizer_all.zero_grad()
                total_loss.backward()    
                # self.optimizer_all.step()
                self.optimizer_st.step()
                self.optimizer_se.step()    
                print("==update of ThetaSE and ThetaST==")  
                                                    
                print("----------batch {} training end----------".format(batch_num))     
                
                total_loss_accum += total_loss.detach().cpu().numpy()
                total_iters += 1
                    
        ##----first epoch only, compute SVDD center----##
        if self.center == None: ##first epoch only
            full_F_list = torch.cat(F_list)
            self.center = torch.mean(full_F_list, dim=0).detach() #no backpropagation for center
            
            average_sc_loss = -1
            average_dc_loss = -1
            average_ca_loss = -1
            average_total_loss = -1
        ##----if not first epoch, compute averaged SVDD loss ----##
        else:
            average_sc_loss = sc_loss_accum/total_iters
            average_dc_loss = dc_loss_accum/total_iters
            average_ca_loss = ca_loss_accum/total_iters
            average_total_loss = total_loss_accum/total_iters

        return average_sc_loss,average_dc_loss,average_ca_loss,average_total_loss
    

    
    def test(self, test_loader):
        print("\n++++++++++++++++trainers.py++++++++++++++++")
        print("----------test()----------")
        self.se_model.eval()
        
        with torch.no_grad():

            prediction_list = []
            for batch in test_loader:
                
                # -----------------------------------------------------------------
                # Step1. use a GIN + readout to learn SEMANTIC FEATURES for source domain (ThetaSE)          
                # -----------------------------------------------------------------
                test_embeddings = self.se_model(batch)                
                mean_test_embeddings = [torch.mean(emb, dim=0) for emb in test_embeddings]


                # -----------------------------------------------------------------
                # Step2. construct KNN graph for target domain          
                # -----------------------------------------------------------------
                import pandas as pd
                from sklearn.neighbors import KDTree 
                mean_test_embeddings_list = [a.tolist() for a in mean_test_embeddings]
                df_mean_test_embeddings = pd.DataFrame.from_records(mean_test_embeddings_list)
                
                ##define attributes for all node
                df_knn_attributes_test = df_mean_test_embeddings
                df_knn_attributes_list_test = df_knn_attributes_test.values.tolist()
                
                ##define edges 
                kdt_test = KDTree(df_knn_attributes_test, leaf_size=30, metric='euclidean')
                n_neighbours = 5
                df_knn_edges_test = kdt_test.query(df_knn_attributes_list_test, k=n_neighbours, return_distance=False)
                df_knn_edges_list_test = []
                for i in range(0, len(df_knn_edges_test)):
                    for j in range(0, n_neighbours):
                        df_knn_edges_list_test.append([i,df_knn_edges_test[i][j]])
                df_knn_edges_list_test = [list(i) for i in zip(*df_knn_edges_list_test)] ##transpose it
                
                ##define KNN graph 
                test_x = torch.tensor(df_knn_attributes_list_test, dtype=torch.float)
                test_edge_index = torch.tensor(df_knn_edges_list_test, dtype=torch.long)
                test_st_data = Data(x=test_x, edge_index=test_edge_index)
                
                
                # -----------------------------------------------------------------
                # Step3. use GIN2 to learn STRUCTURE FEATURES for test domain (ThetaST)          
                # -----------------------------------------------------------------
                test_st_loader = DataLoader([test_st_data], batch_size=1, shuffle=True, pin_memory=True, num_workers=0)
                
                for my_batch in test_st_loader:
                    test_st_embeddings = self.st_model(my_batch)
            
                test_st_embeddings = test_st_embeddings[0]

                # -----------------------------------------------------------------
                # Step4. concatenate the SEMANTIC FEATURES and STRUCTURE FEATURES for test domain          
                # -----------------------------------------------------------------
                
                test_concat_tensor = [torch.cat([a,b], dim=0) for a, b in zip(mean_test_embeddings, test_st_embeddings)]
    
                target_test = torch.stack(test_concat_tensor) 
                # -----------------------------------------------------------------
                # Step5. using SVDD center to predict labels and save prediction results
                # ----------------------------------------------------------------- 
                test_predict_results = torch.sum((target_test - self.center)**2, dim=1).cpu()
                                                                                         
                prediction_list.append(test_predict_results)

           
            labels = torch.cat([batch.y for batch in test_loader])
            preds = torch.cat(prediction_list)
            # print(labels)
            # print(preds)

            ap = average_precision_score(y_true= labels, y_score= preds, average = None, pos_label= 1, sample_weight= None)
            roc_auc = roc_auc_score(y_true= labels, y_score= preds, average = None,
                                    sample_weight= None, max_fpr = None, 
                                    multi_class = 'raise', labels =None)

            return ap, roc_auc, preds, labels


# =============================================================================
# Step 4: Define three GNN models
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_mean_pool


#####build GIN model

class GIN(nn.Module):
    """
    Note: batch normalization can prevent divergence maybe, take care of this later. 
    """
    def __init__(self,  nfeat, nhid, nlayer, dropout=0, act=ReLU(), bias=False, **kwargs):
        super(GIN, self).__init__()
        self.norm = BatchNorm1d
        self.nlayer = nlayer
        self.act = act
        self.transform = Sequential(Linear(nfeat, nhid), self.norm(nhid))
        self.pooling = global_mean_pool
        self.dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList()
        self.nns = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(nlayer):
            self.nns.append(Sequential(Linear(nhid, nhid, bias=bias), 
                                       act, Linear(nhid, nhid, bias=bias)))
            self.convs.append(GINConv(self.nns[-1]))
            self.bns.append(self.norm(nhid))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.transform(x) # weird as normalization is applying to all ndoes in database
        
        # can I also record the distance to center, which is the variance?
        for i in range(self.nlayer):
            x = self.dropout(x)
            x = self.convs[i](x, edge_index)
            x = self.act(x)
            x = self.bns[i](x)

        emb_list = []
        for g in range(data.num_graphs):
            emb = x[data.batch==g]
            emb_list.append(emb)
        #graph_embeds = torch.stack(graph_embeds)

        return emb_list

#####build GIN2 model

class GIN2(nn.Module):
    """
    Note: batch normalization can prevent divergence maybe, take care of this later. 
    """
    def __init__(self,  nfeat, nhid, nlayer, dropout=0, act=ReLU(), bias=False, **kwargs):
        super(GIN2, self).__init__()
        self.norm = BatchNorm1d
        self.nlayer = nlayer
        self.act = act
        self.transform = Sequential(Linear(nfeat, nhid), self.norm(nhid))
        self.pooling = global_mean_pool
        self.dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList()
        self.nns = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(nlayer):
            self.nns.append(Sequential(Linear(nhid, nhid, bias=bias), 
                                       act, Linear(nhid, nhid, bias=bias)))
            self.convs.append(GINConv(self.nns[-1]))
            self.bns.append(self.norm(nhid))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.transform(x) # weird as normalization is applying to all ndoes in database
        
        # can I also record the distance to center, which is the variance?
        for i in range(self.nlayer):
            x = self.dropout(x)
            x = self.convs[i](x, edge_index)
            x = self.act(x)
            x = self.bns[i](x)

        emb_list = []
        for g in range(data.num_graphs):
            emb = x[data.batch==g]
            emb_list.append(emb)
        #graph_embeds = torch.stack(graph_embeds)

        return emb_list
  
#####build Source Classififer model
class SourceClassifier(nn.Module):
    def __init__(self, input_size=128, hidden_size=64, output_size=1):
        super(SourceClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.output = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.output(out)
        
        return out
    
#####build Domain Classififer model
class DomainClassifier(nn.Module):
    def __init__(self, input_size=128, hidden_size=64, output_size=1):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.output = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.output(out)
        return out
    
        
        
    


