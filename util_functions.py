import numpy as np
import warnings
import random
from tqdm import tqdm
import os, sys, time
import networkx as nx
import argparse
import scipy.sparse as ssp
from sklearn import metrics
import warnings
import multiprocessing as mp
import torch
from torch_geometric.transforms import LineGraph
from torch_geometric.data import Data

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in cast")
    

class GNNGraph(object):
    def __init__(self, g, label, node_features, pair_feature=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a numpy array of continuous node features
        '''
        self.num_nodes = node_features.shape[0] 
        self.label = label
        self.node_features = node_features  # numpy array (node_num * feature_dim)
        self.pair_feature = pair_feature
        self.degs = list(dict(g.degree).values())

        if len(g.edges()) != 0:
            x, y = list(zip(*g.edges()))
            self.num_edges = len(x)        
            self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
            self.edge_pairs[:, 0] = x
            self.edge_pairs[:, 1] = y
            self.edge_pairs = self.edge_pairs.flatten()
        else:
            self.num_edges = 0
            self.edge_pairs = np.array([])
        
        # see if there are edge features
        self.edge_features = None
        if nx.get_edge_attributes(g, 'features'):  
            # make sure edges have an attribute 'features' (1 * feature_dim numpy array)
            edge_features = nx.get_edge_attributes(g, 'features')
            assert(type(list(edge_features.values())[0]) == np.ndarray) 
            # need to rearrange edge_features using the e2n edge order
            edge_features = {(min(x, y), max(x, y)): z for (x, y), z in list(edge_features.items())}
            keys = sorted(edge_features)
            self.edge_features = []
            for edge in keys:
                self.edge_features.append(edge_features[edge])
                self.edge_features.append(edge_features[edge])  # add reversed edges
            self.edge_features = np.concatenate(self.edge_features, 0)

def links2subgraphs(A, train_pos, train_neg, test_pos, test_neg, val_pos, val_neg, h=1, max_nodes_per_hop=None, node_information=None, window_size=50, mp_flag=None):
    # extract enclosing subgraphs

    '''Multi-process running'''
    def helper_mp(A, links, g_label, flag=None):
        # the new parallel extraction code
        start = time.time()
        pool = mp.Pool(mp.cpu_count())
        if flag:
            results = pool.map_async(parallel_worker, [((i, j), A, h, max_nodes_per_hop, node_information, window_size) for i, j in zip(links[0], links[1])])
        else:
            results = pool.map_async(parallel_worker, [((i, j), A, h, max_nodes_per_hop, node_information, window_size) for i, j in zip(links[0], links[1])])
        remaining = results._number_left # type: ignore
        pbar = tqdm(total=remaining)
        while True:
            pbar.update(remaining - results._number_left) # type: ignore
            if results.ready(): break
            remaining = results._number_left # type: ignore
            time.sleep(1)
        results = results.get()
        pool.close()
        pbar.close()
    
        g_list = [GNNGraph(g, g_label, n_features, n_pair) for g, n_features, n_pair in results]
        end = time.time()
        print("Time eplased for subgraph extraction: {}s".format(end-start))
        return g_list

    '''Single-process running'''
    def helper_sp(A, links, g_label, flag=None):
        g_list = []
        start = time.time()
        
        for i, j in tqdm(zip(links[0], links[1]), total=len(links[0]), desc=f"Subgraphs Extraction"):
            if flag:
                g, n_features, n_pair = subgraph_extraction_labeling((i, j), A, h, max_nodes_per_hop, node_information, window_size)
            else:    
                g, n_features, n_pair = subgraph_extraction_labeling((i, j), A, h, max_nodes_per_hop, node_information, window_size)
            g_list.append(GNNGraph(g, g_label, n_features, n_pair))
        
        end = time.time()
        print(f"Time elapsed for subgraph extraction: {end - start:.2f}s")
        return g_list

    print('Enclosing subgraph extraction begins...')
    if mp_flag is not None:
        train_graphs = helper_mp(A, train_pos, 1) + helper_mp(A, train_neg, 0)
        test_graphs = helper_mp(A, test_pos, 1, flag=1) + helper_mp(A, test_neg, 0, flag=1)
        val_graphs = helper_mp(A, val_pos, 1) + helper_mp(A, val_neg, 0)
    else:
        train_graphs = helper_sp(A, train_pos, 1) + helper_sp(A, train_neg, 0)
        test_graphs = helper_sp(A, test_pos, 1, flag=1) + helper_sp(A, test_neg, 0, flag=1)
        val_graphs = helper_sp(A, val_pos, 1) + helper_sp(A, val_neg, 0)

    return train_graphs, test_graphs, val_graphs


def parallel_worker(x):
    return subgraph_extraction_labeling(*x)
    

def subgraph_extraction_labeling(ind, A, h=1, max_nodes_per_hop=None, node_information=None, window_size=50):
    # extract the h-hop enclosing subgraph around link 'ind'
    dist = 0
    nodes = set([ind[0], ind[1]])       
    visited = set([ind[0], ind[1]])     
    fringe = set([ind[0], ind[1]])      
    nodes_dist = [0, 0]                 
    for dist in range(1, h+1):
        fringe = neighbors(fringe, A)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes.union(fringe)
        nodes_dist += [dist] * len(fringe)
    # move target nodes to top
    nodes.remove(ind[0])
    nodes.remove(ind[1])
    nodes = [ind[0], ind[1]] + list(nodes) 
    subgraph = A[nodes, :][:, nodes]    
    # get node features
    features = []
    if node_information is not None:
        features = node_information[nodes]
        
        geneA_feature, geneB_feature = node_information[ind[0], :], node_information[ind[1], :]
        geneA_feature = torch.tensor(geneA_feature)
        geneB_feature = torch.tensor(geneB_feature)
        pair_feature = gene_pair_feature_generate(geneA_feature, geneB_feature, window_size)
        
    # construct nx graph
    g = nx.from_scipy_sparse_matrix(subgraph)
    # remove link between target nodes
    if not g.has_edge(0, 1):                
        g.add_edge(0, 1)
    return g, features, pair_feature


def neighbors(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    res = set()
    for node in fringe:
        nei, _, _ = ssp.find(A[:, node])    
        nei = set(nei)
        res = res.union(nei)
    return res


def to_linegraphs(batch_graphs):
    torch.cuda.empty_cache()
    graphs = []

    pbar = tqdm(batch_graphs, unit='iteration')
    for graph in pbar:
        edges = graph.edge_pairs
        information = graph.node_features
        pair_feature = graph.pair_feature

        edge_feas = torch.tensor(information, dtype=torch.float32)
        edges, feas = to_undirect(edges, edge_feas)     
        edges = torch.tensor(edges)

        # Line graph transformation
        data = Data(edge_index=edges, edge_attr=feas)
        data.num_nodes = graph.num_nodes
        data = LineGraph()(data)
        data['y'] = torch.tensor([graph.label])   
        data['pair_feature'] = pair_feature
        data.num_nodes = graph.num_edges
        graphs.append(data)
    return graphs

    
def to_undirect(edges, edge_fea):
    edges = np.reshape(edges, (-1,2 ))
    sr = np.array([edges[:,0], edges[:,1]], dtype=np.int64)
    fea_s = edge_fea[sr[0,:], :]
    fea_s = fea_s.repeat(2,1)
    fea_r = edge_fea[sr[1,:], :]
    fea_r = fea_r.repeat(2,1)
    fea_body = torch.cat([fea_s, fea_r], 1)
    rs = np.array([edges[:,1], edges[:,0]], dtype=np.int64)
    return np.concatenate([sr, rs], axis=1), fea_body


def gene_pair_feature_generate(TF_feature, Target_feature, window_size):
    TF_feature = TF_feature.unsqueeze(0)
    Target_feature = Target_feature.unsqueeze(0)
    temp = []
    for k in range(0, TF_feature.shape[1], window_size):
        subset1 = TF_feature[:, k:k + window_size]
        subset2 = Target_feature[:, k:k + window_size]
        concatenated_tensor = torch.cat((subset1, subset2), dim=1)
        if concatenated_tensor.shape[1] == window_size * 2:
            temp.append(concatenated_tensor.unsqueeze(1))
    gene_pair_feature = torch.cat(temp, dim=1)
    return gene_pair_feature
