"""
#Lab code of ARMET

"""

import warnings
warnings.filterwarnings("ignore")

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import pickle
import argparse
from types import SimpleNamespace

from matplotlib import rcParams
rcParams.update({'figure.autolayout': False})

from DataLoader import create_loaders, MyTrainer, HPSelection, GIN, GIN2, SourceClassifier, DomainClassifier


for DataName in ["BGL","THUNDERBIRD","HDFS","SPIRIT"]:
    
    ##--------------------------------------------
    ##Step 1. first clear all files under the /processed/~ and /raw/~directory
    ##--------------------------------------------
    
    import os, shutil
    folder = r'Data/'+DataName+'/processed'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
            
    folder = r'Data/'+DataName+'/raw'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
            
    ##--------------------------------------------
    ##Step 2. copy all files from a directory to another
    ##--------------------------------------------       
    
    import shutil
    import os
     
    # path to source directory
    src_dir = r'Data/'+DataName+'/Graph/raw/'
     
    # path to destination directory
    dest_dir = r'Data/'+DataName+'/raw/'
     
    # getting all the files in the source directory
    my_files = os.listdir(src_dir)
     
    for file_name in my_files:
        print(file_name)
        print(type(dest_dir))
        src_file_name = src_dir + file_name
        dest_file_name = dest_dir + file_name
        shutil.copy(src_file_name, dest_file_name)
        
##--------------------------------------------
##Step 3. define a function to run experiments
##--------------------------------------------          
        
def run_test(
    source_data = "Letter-low", 
    target_data = "Letter-high",
    test_data = "Letter-high", 
    data_seed=1213, 
    epochs=100, 
    model_seed=0, 
    num_layers=1, 
    sc_weight=0.1,
    dc_weight=1,
    ca_weight=0.01,
    device=0,
    aggregation="Mean", #We can choose it from {"Mean", "Max", "Sum"}
    bias=False,
    hidden_dim=64,
    lr=0.1,
    weight_decay=1e-5,
    batch = 64
    ):

    device = torch.device("cuda:" + str(device)) if torch.cuda.is_available() else torch.device("cpu")
    
    # =============================================================================
    # Step1. load data using predefined script dataloader.py
    # we should define this function by ourself
    # =============================================================================
    
    source_loader, target_loader, test_loader, num_features, source_dataset, target_dataset, test_dataset  = create_loaders(source_data_name=source_data,
                                                                                                                            target_data_name=target_data,
                                                                                                                            test_data_name=test_data,
                                                                                                                            batch_size=batch,
                                                                                                                            is_val = False)
    ##----set seeds for cuda----
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)
        
    # =============================================================================
    # Step2. train a GIN model with given parameters
    # =============================================================================
    
    ##----setting paramters----
    se_model = GIN(nfeat = num_features, nhid=hidden_dim, nlayer=num_layers, bias=bias) ##this one can only handle undirected graphs
    
    ##we use the output of se_model as the input of st_model
    st_model = GIN2(nfeat = hidden_dim, nhid=hidden_dim, nlayer=num_layers, bias=bias) ##this one can only handle undirected graphs

    source_classifier = SourceClassifier(input_size = hidden_dim*2, hidden_size=hidden_dim, output_size=1) 
    
    domain_classifier = DomainClassifier(input_size = hidden_dim*2, hidden_size=hidden_dim, output_size=1) 

    ##----important paramter 0----##
    ##the learning rate, weight decay hyperparameter are given here
    
    optimizer_se = optim.SGD(se_model.parameters(), lr=lr, weight_decay=weight_decay) 
    optimizer_st = optim.SGD(st_model.parameters(), lr=lr, weight_decay=weight_decay)   
    optimizer_dc = optim.SGD(domain_classifier.parameters(), lr=lr, weight_decay=weight_decay)

    trainer = MyTrainer(source_data_name=source_data,
                        target_data_name=target_data,
                        se_model=se_model,
                        st_model=st_model,
                        source_classifier = source_classifier,
                        domain_classifier = domain_classifier,
                        optimizer_se = optimizer_se,
                        optimizer_st = optimizer_st, 
                        optimizer_dc = optimizer_dc,
                        sc_weight = sc_weight,
                        dc_weight = dc_weight,
                        ca_weight = ca_weight,
                        device=device)
    
    ##----starting training----
    for epoch in range(epochs+1):
        ##We need to carefully define the train() function        
        trainer.train(source_data_name=source_data, 
                      target_data_name=target_data, 
                      source_loader= source_loader,
                      target_loader = target_loader, 
                      sc_weight = sc_weight,
                      dc_weight = dc_weight,
                      ca_weight = ca_weight,
                      TSNE_plot = False)            
        
        ap, roc_auc, dists, labels = trainer.test(test_loader=test_loader)

        
    return ap, roc_auc


def run_hp_selection(
    source_data = "Letter-low",
    target_data = "Letter-high", 
    test_data = "Letter-high", 
    data_seed=1213, 
    epochs=100, 
    model_seed=0, 
    num_layers=1, 
    sc_weight=0.1,
    dc_weight=1,
    ca_weight=0.01,
    device=0,
    aggregation="Mean", #We can choose it from {"Mean", "Max", "Sum"}
    bias=False,
    hidden_dim=64,
    lr=0.1,
    weight_decay=1e-5,
    batch = 64
    ):

    device = torch.device("cuda:" + str(device)) if torch.cuda.is_available() else torch.device("cpu")
    
    # =============================================================================
    # Step1. load data using predefined script dataloader.py
    # we should define this function by ourself
    # =============================================================================
    
    source_loader, target_loader, val_loader, num_features, source_dataset, target_dataset, test_dataset  = create_loaders(source_data_name=source_data,
                                                                                                                            target_data_name=target_data,
                                                                                                                            test_data_name=test_data,
                                                                                                                            batch_size=batch,
                                                                                                                            is_val = True)
    ##----set seeds for cuda----
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)
        
    # =============================================================================
    # Step2. train a GIN model with given parameters
    # =============================================================================
    
    ##----setting paramters----
    se_model = GIN(nfeat = num_features, nhid=hidden_dim, nlayer=num_layers, bias=bias) ##this one can only handle undirected graphs
    
    ##we use the output of se_model as the input of st_model
    st_model = GIN2(nfeat = hidden_dim, nhid=hidden_dim, nlayer=num_layers, bias=bias) ##this one can only handle undirected graphs

    source_classifier = SourceClassifier(input_size = hidden_dim*2, hidden_size=hidden_dim, output_size=1) 
    
    domain_classifier = DomainClassifier(input_size = hidden_dim*2, hidden_size=hidden_dim, output_size=1) 

    ##----important paramter 0----##
    ##the learning rate, weight decay hyperparameter are given here
    
    ##https://blog.csdn.net/qq_27292549/article/details/96860093
    optimizer_se = optim.SGD(se_model.parameters(), lr=lr, weight_decay=weight_decay) 
    optimizer_st = optim.SGD(st_model.parameters(), lr=lr, weight_decay=weight_decay)   
    optimizer_dc = optim.SGD(domain_classifier.parameters(), lr=lr, weight_decay=weight_decay)

    trainer = HPSelection(source_data_name=source_data,
                          target_data_name=target_data,
                          se_model=se_model,
                          st_model=st_model,
                          source_classifier = source_classifier,
                          domain_classifier = domain_classifier,
                          optimizer_se = optimizer_se,
                          optimizer_st = optimizer_st, 
                          optimizer_dc = optimizer_dc,
                          sc_weight = sc_weight,
                          dc_weight = dc_weight,
                          ca_weight = ca_weight,
                          device=device)
    
    ##----starting training----
    for epoch in range(epochs+1):
        ##We need to carefully define the train() function        
        trainer.train(source_data_name=source_data, 
                      target_data_name=target_data, 
                      source_loader= source_loader,
                      target_loader = target_loader, 
                      sc_weight = sc_weight,
                      dc_weight = dc_weight,
                      ca_weight = ca_weight,
                      TSNE_plot = False)            
        
        ap, roc_auc, dists, labels = trainer.test(test_loader=val_loader)


    return ap, roc_auc

# =============================================================================
# Step 4: define a parser
# The argparse module makes it easy to write user-friendly command-line interfaces. 
# The program defines what arguments it requires, and argparse will figure out 
# how to parse those out of sys.argv
# =============================================================================
        
        
import os
from pathlib import Path
for my_source_name in ["BGL", "SPIRIT", "HDFS", "THUNDERBIRD"]:
    for my_target_name in ["BGL", "SPIRIT", "HDFS", "THUNDERBIRD"]:            
        for my_is_source in ['False']:
            for run in ['1']:
                if my_source_name == my_target_name:
                    pass
                else:
                    folder = my_source_name + "_" + my_target_name

                    if not Path(folder).exists():
                        print("Creating folder for working directory")
                        Path(folder).mkdir()

                    my_source_data = my_source_name
                    my_target_data = my_target_name
                    my_test_data = my_target_name
                    

                    seed_value = int(run)

                    folder = os.path.join(folder, "my_is_source_" + str(my_is_source))
                    if not Path(folder).exists():
                        print("Creating folder for working directory")
                        Path(folder).mkdir()

                    folder = os.path.join(folder, run)
                    if not Path(folder).exists():
                        print("Creating folder for working directory")
                        Path(folder).mkdir()

                    if "Letter" in my_source_name:
                        my_batch_size = 128 ##128 by default
                        my_epochs = 100 ##100 by default
                        my_hidden_dim = 64 ## 64 by default
                        my_GIN_layers = 2 ## 2 by default
                        my_lr = 0.01 ##0.01 by default
                        my_weight_decay  = 1e-4
                    else:
                        my_batch_size = 512 ##512 by default
                        my_epochs = 100 ##100 by default
                        my_hidden_dim = 64 ##64 by default
                        my_GIN_layers = 2 ## 2 by default
                        my_lr = 0.01 ##0.01 by default
                        my_weight_decay  = 1e-4 ##1e-4 by default


                    parser = argparse.ArgumentParser(description='CDGLAD:')

                    ##----important paramter 1----##
                    parser.add_argument('--source_data', default= my_source_data) 
                    parser.add_argument('--target_data', default= my_target_data) 
                    parser.add_argument('--test_data', default= my_test_data) 

                    parser.add_argument('--batch', type=int, default= my_batch_size,
                                        help='batch size (default: 64)')
                    parser.add_argument('--data_seed', type=int, default=421,
                                        help='seed to split the inlier set into train and test (default: 1213)')
                    parser.add_argument('--device', type=int, default=0,
                                        help='which gpu to use if any (default: 0)')

                    parser.add_argument('--epochs', type=int, default= my_epochs, 
                                        help='number of epochs to train (default: 150)')
                    parser.add_argument('--hidden_dim', type=int, default= my_hidden_dim,
                                        help='number of hidden units (default: 64)')
                    parser.add_argument('--layers', type=int, default= my_GIN_layers,
                                        help='number of hidden layers (default: 2)')

                    ##----important paramter 2----##
                    parser.add_argument('--bias', action="store_true", default = False,
                                                        help='Whether to use bias terms in the GNN.')

                    parser.add_argument('--aggregation', type=str, default="Mean", choices=["Max", "Mean", "Sum"],
                                        help='Type of graph level aggregation (default: Mean)')

                    parser.add_argument('--use_config', action="store_true",
                                                        help='Whether to use configuration from a file')
                    parser.add_argument('--config_file', type=str, default="configs/config.txt",
                                        help='Name of configuration file (default: configs/config.txt)')


                    parser.add_argument('--lr', type=float, default=my_lr,
                                        help='learning rate (default: 0.01)')
                    parser.add_argument('--weight_decay', type=float, default= my_weight_decay,
                                        help='weight_decay constant lambda (default: 1e-4)')
                    parser.add_argument('--model_seed', type=int, default=seed_value, 
                                        help='Model seed (default: 0)')

                    # =============================================================================
                    # Step 5:  configure paramters
                    # store each paramter as an individual list since we want to do model selection
                    # =============================================================================
                    args = parser.parse_args()

                    lrs = [args.lr]
                    weight_decays = [args.weight_decay]
                    layercounts = [args.layers]
                    model_seeds = [args.model_seed]


                    ##larger search ranges if time allows
                    ##-----------------------------------
                    
                    epoch_vec = [100,200,300,400]
                    lambda_sc_vec  = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
                    lambda_dc_vec= [0.00001,0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
                    lambda_ca_vec = [0.00001,0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

                    # =============================================================================
                    # Step 6. we store all model candidates by traversing all parameter value lists
                    # =============================================================================

                    ##use a dictionary to store model hyperparameters for different model candidates

                    HP_results = []

                    ##rewrite this part to do HP selection:
                        
                    for lr in lrs:
                        for weight_decay in weight_decays:
                            for model_seed in model_seeds:
                                for layercount in layercounts:
                                    for lambda_sc in lambda_sc_vec:
                                        for lambda_dc in lambda_dc_vec:
                                            for lambda_ca in lambda_ca_vec:
                                                for epoch_nb in epoch_vec:
                                                    my_ap, my_roc = run_hp_selection(
                                                        source_data=args.source_data,
                                                        target_data=args.target_data,
                                                        test_data=args.test_data,
                                                        data_seed=args.data_seed,
                                                        epochs=epoch_nb,     # HYPERPARAMETER
                                                        model_seed=model_seed, 
                                                        num_layers=layercount, 
                                                        sc_weight=lambda_sc, # HYPERPARAMETER
                                                        dc_weight=lambda_dc, # HYPERPARAMETER
                                                        ca_weight=lambda_ca, # HYPERPARAMETER
                                                        device=args.device,
                                                        aggregation=args.aggregation,
                                                        bias=args.bias,
                                                        hidden_dim=args.hidden_dim,
                                                        lr=lr,  
                                                        weight_decay=weight_decay, 
                                                        batch=args.batch
                                                    )
                                                    HP_results.append([lr,weight_decay,layercount,lambda_sc,lambda_dc,lambda_ca,epoch_nb,my_ap,my_roc])
                    import pandas as pd
                    HP_results_df = pd.DataFrame(HP_results, columns=["lr","weight_decay","layercount","lambda_sc","lambda_dc","lambda_ca","epoch_nb","my_ap","my_roc"])


                    # =============================================================================
                    # Step 7. use the best hyperparamter lists to test
                    # =============================================================================

                    sorted_HP_results_df = HP_results_df.sort_values(by='my_roc', ascending=False)
                    opt_hp_vec = sorted_HP_results_df.iloc[0].tolist()

                    opt_lr = opt_hp_vec[0]
                    opt_weight_decay = opt_hp_vec[1]
                    opt_layercount = int(opt_hp_vec[2])
                    opt_lambda_sc = opt_hp_vec[3]
                    opt_lambda_dc = opt_hp_vec[4]
                    opt_lambda_ca = opt_hp_vec[5]
                    opt_epochs = int(opt_hp_vec[6])

                    best_ap, best_roc = run_test(source_data=args.source_data,
                                                target_data=args.target_data,
                                                test_data=args.test_data,
                                                data_seed=args.data_seed,
                                                epochs=opt_epochs,     # HYPERPARAMETER
                                                model_seed=model_seed, # HYPERPARAMETER
                                                num_layers=opt_layercount, 
                                                sc_weight=opt_lambda_sc, # HYPERPARAMETER
                                                dc_weight=opt_lambda_dc, # HYPERPARAMETER
                                                ca_weight=opt_lambda_ca, # HYPERPARAMETER
                                                device=args.device,
                                                aggregation=args.aggregation,
                                                bias=args.bias,
                                                hidden_dim=args.hidden_dim,
                                                lr=opt_lr,  # HYPERPARAMETER
                                                weight_decay=opt_weight_decay, # HYPERPARAMETER
                                                batch=args.batch 

                                                )

                    with open(os.path.join(folder,'result.txt'), 'w') as file:
                        file.write("average_loss:" + str(0) + '\n')
                        file.write("roc_auc:" + str(best_roc) + '\n')
                        file.write("pr:" + str(best_ap) + '\n')
                        file.write("f1:" + str(0) + '\n')                    
