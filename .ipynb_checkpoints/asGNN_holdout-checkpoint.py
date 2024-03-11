import os
import ray
import time
import copy
import torch
import pickle
import argparse
import itertools
import GNN_core
import smoothing_based_optimizer
import numpy as np
from torch_geometric.loader import DataLoader


parser = argparse.ArgumentParser(description="Simulate a Affinity training GNN with the appropriate hyperparameters.")
parser.add_argument('-d','--dataset', required=True, help='the protein dataset')
parser.add_argument('--graph_path', required=True, help='path to the graph files')
parser.add_argument('-r','--partition_ratio', required=False, type=str, help="governs the ration of partition sizes in the training, validation, and test sets. a list of the form [train, val, test]", default="12:4:7")
parser.add_argument('-e','--epochs', required=False, help='number of training epochs', default='201')
parser.add_argument('-nc','--num_layers', required=False, help='number of layers', default='3')
parser.add_argument('-p','--patience', required=False, type=int, help='upper limit for the patience counter used in validation', default=60)
parser.add_argument('-b','--batch_size', required=False, type=int, help='batch size for training, testing and validation', default=30)
parser.add_argument('-l_set','--learning_rate_set', required=False, type=str, help='initial learning rate', default="0.001,0.005,0.01,0.05")
parser.add_argument('-m','--model_type', required=False, type=str, help='the underlying model of the neural network', default='GCN')
parser.add_argument('--corr', help='add correlation loss', action='store_true')
parser.add_argument('-hc_set','--hidden_channel_set', required=False, type=str, help='width of hidden layers', default="25,50,100,250,500")
parser.add_argument('--meta_feature_dim', required=False, help='the output size of the first linear layer or the input dimension of the clustering algorithm', default='same')
#parser.add_argument('--cluster_params', required=False, help='p_factor, C_sp and C_degree', default="4,0.8,0.1")
parser.add_argument('--p_factor_set', required=False, help='p_factor', default="1,2,3,4,5,6")
parser.add_argument('--C_sp_set', required=False, help='p_factor, C_sp and C_degree', default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8")
parser.add_argument('--transform', required=False, help='transform', default="offset")
parser.add_argument('--sample_size', required=False, help='sample size for optimization', type=int, default=15)
parser.add_argument('--initial_mu', required=False, help='the starting point of the gaussian distribution mu for smoothing based optimization', default="identity")
parser.add_argument('--initial_sigma', required=False, type=float, help='the starting point of the gaussian distribution sigma for smoothing based optimization', default=3.)
parser.add_argument('-o','--params_storage_path',required=False,help='path to store the model params',default=None)

args = parser.parse_args()
HE_dataset=args.dataset
HE_graph_path=args.graph_path
n_epochs=args.epochs
arch=args.model_type
ratio = args.partition_ratio.split(":")
ratio = [float(entry) for entry in ratio]
batch_size=args.batch_size
num_layers=args.num_layers
corr = args.corr

lr_set=[float(lr) for lr in args.learning_rate_set.split(',')]
hidden_channel_set=[int(hc) for hc in args.hidden_channel_set.split(',')]

meta_feature_dim=args.meta_feature_dim
p_factor_set=[float(p_factor) for p_factor in args.p_factor_set.split(',')]
C_sp_set=[float(C_sp) for C_sp in args.C_sp_set.split(',')]
transform=args.transform

output_path=args.params_storage_path


### load HE sections
patients = []
HE_sections=[]
with open(HE_dataset, 'r') as file:
    content = file.read()
for line in content.splitlines():
    line=np.array(list(line.split(' ')))
    HE_sections.append(line[0])
    patients.append(line[0].split('_')[0])


if __name__ == '__main__':

    torch.manual_seed(12345)
    
    graph_dataset=[]

    for HE_section_id, HE_section in enumerate(HE_sections):
        if os.path.exists(str(HE_graph_path) + '/' + str(HE_section) + '.nx'):
            with open(str(HE_graph_path) + '/' + str(HE_section) + '.nx', 'rb') as f:
                G = pickle.load(f)
            G.section_idx = torch.tensor(HE_section_id, dtype=torch.long)
            G.x = G.x.to(torch.float32)
            graph_dataset.append(G)

    assert(ratio[0]+ratio[1]+ratio[2] == len(np.unique(patients)))
    patients = np.array(patients)[sorted(np.unique(patients, return_counts=True, return_index=True)[1])]
    if np.where(patients == 'BT23901')[0] < ratio[0]:
        part1 = int((ratio[0]-1)*3+2)
        part2 = int(part1 + ratio[1]*3)
        part3 = int(part2 + ratio[2]*3)
    elif np.where(patients == 'BT23901')[0] > (ratio[0] + ratio[1]):
        part1 = int(ratio[0]*3)
        part2 = int(part1 + ratio[1]*3)
        part3 = int(part2 + (ratio[2]-1)*3+2)
    else:
        part1 = int((ratio[0])*3)
        part2 = int(part1 + (ratio[1]-1)*3+2)
        part3 = int(part2 + ratio[2])

    num_node_features=len(graph_dataset[0].x[0])
    if str(meta_feature_dim)=='same':
        meta_feature_dim=len(graph_dataset[0]['x'][0])
    else:
        meta_feature_dim=int(meta_feature_dim)

    if os.path.exists(os.path.join(output_path, 'GNN_hyperparams_tuning.npz')): 
        results = np.load(os.path.join(output_path, 'GNN_hyperparams_tuning.npz'))
        GNN_params_set = results['GNN_params_set']
        val_loss_lst = results['val_loss_lst']
        GNN_params = GNN_params_set[np.argmin(val_loss_lst)]
        hidden_channels = int(GNN_params[0])
        lr = float(GNN_params[1])
        print(f'best learning rate: {lr} and best number of hidden channels: {hidden_channels}')
        best_model_path = os.path.join(output_path, 'GNN_hyperparams_tunning_best_model.pt')
    else:
        if not os.path.exists(os.path.join(output_path, 'GNN_hyperparams_tunning')):
            os.makedirs(os.path.join(output_path, 'GNN_hyperparams_tunning'))
        
        # Find best hyperparameters for GTN backbone 
        num_classes = 250
    
        train_dataset = graph_dataset[:part1]
        val_dataset = graph_dataset[part1:part2]
        test_dataset = graph_dataset[part2:]
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        train_loss_lst = []; train_mse_lst = []; train_pcc_lst = []
        val_loss_lst = []; val_mse_lst = []; val_pcc_lst = []
        test_loss_lst = []; test_mse_lst = []; test_pcc_lst = []
        best_val_loss = np.Inf
        best_model = None
        best_lr = None
        best_hidden_channels = None
        time3=time.time()
        for hidden_channels in hidden_channel_set:
            for lr in lr_set:
                if arch == 'GCN':
                    model = GNN_core.GCN(hidden_channels,input_dim=meta_feature_dim,num_classes=num_classes,num_layers=num_layers)
                if arch == 'GTN':
                    model = GNN_core.GTN(hidden_channels,input_dim=meta_feature_dim,num_classes=num_classes,num_layers=num_layers)
                if arch == 'GNN':
                    model = GNN_core.GNN(hidden_channels,input_dim=meta_feature_dim,num_classes=num_classes,num_layers=num_layers)
                optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    
                _, val_res, train_res, test_res, trained_model = smoothing_based_optimizer.train_GNN(model=model,train_loader=train_loader, 
                                                                                                            val_loader=val_loader,test_loader=test_loader,
                                                                                                            optimizer=optimizer,n_epochs=n_epochs,patience=args.patience,
                                                                                                            transform=transform,corr=corr)
                this_val_loss = val_res[0]
                
                train_loss_lst.append(train_res[0]); train_mse_lst.append(train_res[1]); train_pcc_lst.append(train_res[2])
                val_loss_lst.append(val_res[0]); val_mse_lst.append(val_res[1]); val_pcc_lst.append(val_res[2])
                test_loss_lst.append(test_res[0]); test_mse_lst.append(test_res[1]); test_pcc_lst.append(test_res[2])
                
                if this_val_loss < best_val_loss:
                    best_val_loss = this_val_loss
                    best_lr = lr
                    best_hidden_channels = hidden_channels
                    best_model=copy.deepcopy(trained_model)

                torch.save(trained_model, os.path.join(output_path, 'GNN_hyperparams_tunning', 'GNN_hyperparams_tunning_model_' + str(hidden_channels) + '_' + str(lr) + '.pt'))
                    
        time4=time.time()
        lr = best_lr
        hidden_channels = best_hidden_channels
        print(f'best learning rate: {lr} and best number of hidden channels: {hidden_channels}')
        print('time for GTN backbone hyperparameter selection:', time4-time3)
        
        GNN_params_set = [list(GNN_params) for GNN_params in list(itertools.product(hidden_channel_set, lr_set))]
        np.savez(os.path.join(output_path, 'GNN_hyperparams_tuning.npz'), GNN_params_set=GNN_params_set, 
                 train_loss_lst=train_loss_lst, val_loss_lst=val_loss_lst, test_loss_lst=test_loss_lst,
                 train_mse_lst=train_mse_lst, val_mse_lst=val_mse_lst, test_mse_lst=test_mse_lst, 
                 train_pcc_lst=train_pcc_lst, val_pcc_lst=val_pcc_lst, test_pcc_lst=test_pcc_lst)
        best_model_path = os.path.join(output_path, 'GNN_hyperparams_tunning_best_model.pt')
        torch.save(best_model, best_model_path)

    # calculate degree and distance matrices
    result_degree=[0]*len(graph_dataset)
    result_distance=[0]*len(graph_dataset)
    
    ray.init()
    BATCHES=len(graph_dataset)
    time3=time.time()
    for i in range(BATCHES):
        result_degree[i]=smoothing_based_optimizer.calc_degree_ray.remote(graph_dataset,i)
        result_distance[i]=smoothing_based_optimizer.calc_distance_ray.remote(graph_dataset,i)
    degree_matrices = ray.get(result_degree)
    distance_matrices = ray.get(result_distance)
    time4=time.time()
    ray.shutdown()
    print('matrices time with ray:',time4-time3)
    
    if os.path.exists(os.path.join(output_path, 'AP_clustering_hyperparams_tuning.npz')): 
        results = np.load(os.path.join(output_path, 'AP_clustering_hyperparams_tuning.npz'))
        cluster_params_set = results['cluster_params_set']
        val_mse_lst = results['val_mse_lst']
        cluster_params = cluster_params_set[np.argmin(val_mse_lst)]
        print(f'best p_factor: {cluster_params[0]} and C_sp: {cluster_params[1]}')
    else:
        # Find best hyperparameters for AP clustering
        ray.init()
        time3=time.time()
        results = []
        cluster_params_set = [list(cluster_params) for cluster_params in list(itertools.product(p_factor_set, C_sp_set, [0]))]
    
        mu=[0.]*(meta_feature_dim*num_node_features+meta_feature_dim)
        for i in range(meta_feature_dim):
            mu[meta_feature_dim+i*num_node_features+i]=1.
        
        for cluster_params in cluster_params_set:
            results.append(smoothing_based_optimizer.calc_loss_onePoint.remote(mu,0,meta_feature_dim,num_node_features,graph_dataset,part1,part2,batch_size,n_epochs,args.patience,degree_matrices,distance_matrices,hidden_channels,num_layers,arch,lr,cluster_params,transform,corr,trained_model_path=best_model_path))
    
        output = ray.get(results)
        train_loss_lst=[np.float_(row[3][0]) for row in output]; train_mse_lst=[np.float_(row[3][1]) for row in output]; train_pcc_lst=[np.float_(row[3][2]) for row in output]
        val_loss_lst=[np.float_(row[2][0]) for row in output]; val_mse_lst=[np.float_(row[2][1]) for row in output]; val_pcc_lst=[np.float_(row[2][2]) for row in output]
        test_loss_lst=[np.float_(row[4][0]) for row in output]; test_mse_lst=[np.float_(row[4][1]) for row in output]; test_pcc_lst=[np.float_(row[4][2]) for row in output]
    
        cluster_params = cluster_params_set[np.argmin(np.array(val_loss_lst))]
        print(f'best p_factor: {cluster_params[0]} and C_sp: {cluster_params[1]}')
        
        time4=time.time()
        ray.shutdown()
        print('time for AP clustering hyperparameter selection (with ray):', time4-time3)
        np.savez(os.path.join(output_path, 'AP_clustering_hyperparams_tuning_complete.npz'), cluster_params_set=cluster_params_set, 
                 train_loss_lst=train_loss_lst, val_loss_lst=val_loss_lst, test_loss_lst=test_loss_lst,
                 train_mse_lst=train_mse_lst, val_mse_lst=val_mse_lst, test_mse_lst=test_mse_lst, 
                 train_pcc_lst=train_pcc_lst, val_pcc_lst=val_pcc_lst, test_pcc_lst=test_pcc_lst)

##############################

    ray.init()
    if args.initial_mu=='identity':
        mu=[0.]*(meta_feature_dim*num_node_features+meta_feature_dim) ##set the initial value to (partial) identity matrix
        for i in range(meta_feature_dim):
            mu[meta_feature_dim+i*num_node_features+i]=1. ##mu[meta_feature_dim+i*num_node_features+i]=1.
    else:
        mu=np.loadtxt(args.initial_mu)

    sigma=args.initial_sigma

    epsilon=0.0001
    t=0
    sigma_list=[]
    val_mse_list=[]; val_pcc_list=[]
    test_mse_list=[]; test_pcc_list=[]
    train_mse_list=[]; train_pcc_list=[]

    # Load last chcekpoint
    if os.path.exists(output_path):
        checkpoints = [checkpoint for checkpoint in os.listdir(output_path) if os.path.isdir(checkpoint) and not checkpoint.startswith('.')]
        if len(checkpoints) != 0 :
            t = max([int(checkpoint.split('t')[1]) for checkpoint in checkpoints]) - 1
            if t != -1:
                mu = np.loadtxt(os.path.join(output_path, 't' + str(t), 'mu.txt'))
                sigma = np.loadtxt(os.path.join(output_path, 't' + str(t), 'sigma.txt'))
            else:
                t = 0
    
    while epsilon<sigma and t<60:
        
        print('\n start with t =',t)
        
        N_sample=args.sample_size
        time1=time.time()
        BATCHES=N_sample
        results=[]

        ### update mu
        for i in range(BATCHES):

            # fix sample 0 for each meta-epoch to be the mean
            if i == 0:
                if t>0:
                    results.append(smoothing_based_optimizer.calc_loss_onePoint.remote(mu,0,meta_feature_dim,num_node_features,graph_dataset,part1,part2,batch_size,n_epochs,args.patience,degree_matrices,distance_matrices,hidden_channels,num_layers,arch,lr,cluster_params,transform,corr))
                else:
                    results.append(smoothing_based_optimizer.calc_loss_onePoint.remote(mu,0,meta_feature_dim,num_node_features,graph_dataset,part1,part2,batch_size,n_epochs,args.patience,degree_matrices,distance_matrices,hidden_channels,num_layers,arch,lr,cluster_params,transform,corr,trained_model_path=best_model_path))
            else:
                results.append(smoothing_based_optimizer.calc_loss_onePoint.remote(mu,sigma,meta_feature_dim,num_node_features,graph_dataset,part1,part2,batch_size,n_epochs,args.patience,degree_matrices,distance_matrices,hidden_channels,num_layers,arch,lr,cluster_params,transform,corr))
                
        output = ray.get(results)

        sample=[np.float_(row[0]) for row in output]
        objective_func=np.array([row[1] for row in output])
        new_mu=smoothing_based_optimizer.update_mu(sample,objective_func)

        if type(new_mu)!=np.ndarray and new_mu==None:
            new_mu=mu

        new_sigma=smoothing_based_optimizer.update_sigma(sample,objective_func,current_mu=new_mu)
        if new_sigma==None:
            new_sigma=sigma
        time2=time.time()
        
        print('calculate all',N_sample,'samples in t =',t,'in time',time2-time1)

        val_res=[row[2] for row in output]; val_mse=[row[2][1] for row in output]; val_pcc=[row[2][2] for row in output]
        train_res=[row[3] for row in output]; train_mse=[row[3][1] for row in output]; train_pcc=[row[3][2] for row in output]
        test_res=[row[4] for row in output]; test_mse=[row[4][1] for row in output]; test_pcc=[row[4][2] for row in output]
        best_model=[row[5] for row in output]
        
        if output_path!=None:
            #smoothing_based_optimizer.store_model_params(output_path,sample_1,objective_func_1,best_model,val_mse,train_mse,test_mse,t)
            smoothing_based_optimizer.store_model_params(output_path,sample,objective_func,best_model,val_res,train_res,test_res,t)

            my_path=str(output_path)+'/t'+str(t)
            with open(my_path+'/mu.txt', 'w') as fp:
                for item in new_mu:
                    fp.write("%s\n" % item)
            with open(my_path+'/sigma.txt', 'w') as fpp:
                fpp.write("%s\n" % new_sigma)
        
        average_val_mse=np.mean(val_mse)
        average_train_mse=np.mean(train_mse)
        average_test_mse=np.mean(test_mse)
        average_val_pcc=np.mean(val_pcc)
        average_train_pcc=np.mean(train_pcc)
        average_test_pcc=np.mean(test_pcc)

        val_mse_list.append(average_val_mse)
        train_mse_list.append(average_train_mse)
        test_mse_list.append(average_test_mse)
        val_pcc_list.append(average_val_pcc)
        train_pcc_list.append(average_train_pcc)
        test_pcc_list.append(average_test_pcc)
        
        print('all test mse:', test_mse)
        print('sigma:', sigma)
        print('average val mse:',average_val_mse)
        print('average val pcc:',average_val_pcc)
        print('average train mse:',average_train_mse)
        print('average train pcc:',average_train_pcc)
        print('average test mse:',average_test_mse)
        print('average test pcc:',average_test_pcc)

        sigma=new_sigma
        mu=new_mu
        t+=1

    ray.shutdown()
    
    print('final result!')
    print('final mu',mu)
    print('all mean val mse',val_mse_list)
    print('all mean test mse',test_mse_list)
    print('all mean val pcc',val_pcc_list)
    print('all mean test pcc',test_pcc_list)
