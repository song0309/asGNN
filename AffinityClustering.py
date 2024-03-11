import numpy as np
from collections import defaultdict
import time
from sklearn.cluster import AffinityPropagation
from torch_geometric.seed import seed_everything
import torch

torch.manual_seed(12345)

def cluster_label2list(label):
    cluster=[[] for _ in range(np.max(label)+1)] 
    for i in range(len(label)):
        cluster[label[i]].append(i)
    return cluster

def calc_degree(graph_dict,index):
    return len(graph_dict[index])

def graph_dictionary(edge_first,edge_second,n_nodes):
    graph_dict = defaultdict(list)

    for i in range(len(edge_first)):
        a, b = edge_first[i], edge_second[i]    
        # Creating the graph as adjacency list
        graph_dict[a].append(b)
        graph_dict[b].append(a)
    if len(graph_dict)!=n_nodes:
        for index in range(n_nodes):
            graph_dict[index]

    return graph_dict

def GWJV_SP(graph_dict):
    max_level=5 #threshold for max distance
    n_nodes=len(graph_dict)
    SP=np.zeros((n_nodes,n_nodes))-1
    for node in range(n_nodes):
        n_level=0
        current_level = [node]
        current_distance = 0
        SP[node][node] = 0
        while len(current_level)>0 and current_distance<max_level:
            current_distance += 1
            next_level = []
            for n in current_level:
                for ne in graph_dict[n]:
                    if SP[node][ne] == -1:
                        next_level.append(ne)
                        SP[node][ne] = current_distance
            current_level = next_level
            n_level+=1
    SP[SP == -1] = max_level
    return SP

def AffinityClustering(x,edges,degree_matrix,distance_matrix,current_batch,physical_params):
    p_factor=physical_params[0].item()
    batch_offset=0
    nodes=x.detach().numpy()
    max_cluster=0
    cluster_label=[]
    for index,my_batch in enumerate(current_batch):
        my_n_nodes=len(my_batch.x)
        my_x=nodes[batch_offset:batch_offset+my_n_nodes]
        
        C_sp=physical_params[1].item()
        C_degree=physical_params[2].item()
        #print('cluster_params: ',p_factor,C_sp,C_degree)
        dis=my_x[:,None]-my_x[None,:]
        feature=np.linalg.norm(dis,axis=2)
        S=-(C_degree*degree_matrix[index]+C_sp*distance_matrix[index]+feature)
        S_median=np.median(S.flatten())
        np.fill_diagonal(S,S_median)
        preference=S_median*p_factor
        af=AffinityPropagation(damping=0.88,max_iter=1600,preference=preference,affinity='precomputed',random_state=0)
        af_label = af.fit_predict(S)
        af_label=af_label+max_cluster
        #print('cluster_size',len(feature)/max(af_label))
        cluster_label=np.append(cluster_label,af_label)
        max_cluster=max_cluster+max(af_label)+1
        batch_offset=batch_offset+my_n_nodes
        # damping 0.88 and max_iter=1600 help to remove all convergence issues
        # current cluster size \approx 30 for S_median*2, 80 for S_median*10, 40 for S_median*5

    edges=edges.numpy().T
    to_remove=[]

    for index,edge in enumerate(edges):
        if cluster_label[edge[0]]!=cluster_label[edge[1]]:
            to_remove.append(index)
    edges=np.delete(edges,obj=to_remove, axis=0)
    t = torch.from_numpy(edges.T)

    return t

def predetermined_cluster(edges,cluster_final):
    edges=edges.numpy().T
    to_remove=[]

    for index,edge in enumerate(edges):
        if cluster_final[edge[0]]!=cluster_final[edge[1]]:
            to_remove.append(index)

    edges=np.delete(edges,obj=to_remove, axis=0)
    t = torch.from_numpy(edges.T)

    return t

def AffinityClustering_oneGraph(x,edges,degree_matrix,distance_matrix,physical_params):
    p_factor=physical_params[0]

    C_sp=physical_params[1]
    C_degree=physical_params[2]

    dis=x[:,None]-x[None,:]
    feature=np.linalg.norm(dis,axis=2)
    S=-(C_degree*degree_matrix+C_sp*distance_matrix+feature)
    S_median=np.median(S.flatten())
    np.fill_diagonal(S,S_median)
    preference=S_median*p_factor
    af=AffinityPropagation(damping=0.88,max_iter=1600,preference=preference,affinity='precomputed',random_state=0)
    af_label = af.fit_predict(S)


    edges=edges.numpy().T
    to_remove=[]

    for index,edge in enumerate(edges):
        if af_label[edge[0]]!=af_label[edge[1]]:
            to_remove.append(index)
    edges=np.delete(edges,obj=to_remove, axis=0)
    t = torch.from_numpy(edges.T)

    return t


def calc_distance_matrix(graph_dict):
    SP=GWJV_SP(graph_dict)
    return SP

def calc_degree_matrix(graph_dict):
    n_nodes=len(graph_dict)
    degree_matrix=np.zeros((n_nodes,n_nodes))
    
    for i in range(n_nodes):
        degree_i=calc_degree(graph_dict,i)
        for j in range(i+1,n_nodes):
            degree_j=calc_degree(graph_dict,j)
            degree_matrix[i][j]=abs(degree_i-degree_j)
            degree_matrix[j][i]=degree_matrix[i][j]
    return degree_matrix
