import numpy as np
from torch_geometric.data import Data, Dataset
import networkx as nx

class CLRSData(Data):
    """A data object for CLRS data."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

#----------------------------------------------------------
# CLASS Graph
import matplotlib.pyplot as plt
class Graph:
    """Generates a graph with n nodes and probability p of edge creation. The graph is stored as an adjacency matrix and a list of edges."""
    def __init__(self, n:int, m:int = None, p:float = None, directed:bool=False, type:str='erdos_renyi'):
        self.n = n # number of nodes
        #self.p = p # probability of edge creation
        if n is None or n<=0:
            raise ValueError('Please provide a valid number of nodes')
        if type == 'erdos_renyi':
            if p is None or p<=0 or p>=1:
                raise ValueError('p must be specified for Erdos-Renyi graph to be between 0 and 1')
            self.graph = nx.erdos_renyi_graph(n, p, directed=directed)

        elif type == 'barabasi_albert':
            if m is None or m<=0 or m>=n:
                raise ValueError('m must be specified for Barabasi-Albert graph and must be positif and <n')
            self.graph = nx.barabasi_albert_graph(n, m, seed=None)

        self.adj = nx.to_numpy_array(self.graph)
        self.edges_indexes = self.get_edges_indexes(self.adj)

    def get_edges_indexes(self, A):
        edge_indexes = [[],[]]
        """Create a 2x1 list of lists to store the indexes of the edges in the adjacency matrix."""
        for i in range(len(A)):
            for j in range(len(A)):
                if A[i][j] == 1:
                    edge_indexes[0].append(i)
                    edge_indexes[1].append(j)
        return edge_indexes

    def draw(self):
        nx.draw(self.graph, with_labels=True)
        plt.show()

    def saveRawFile(self, path):
        """saves the graph as an edge list"""
        nx.write_edgelist(self.graph, path, delimiter='\t', data=False)

#----------------------------------------------------------
# CLASS GraphGenerator
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os
import os.path as osp

class GraphGenerator:
    """Generates a raw dataset of graphs. (.edgelist files)"""
    def __init__(self, n, m=None, p=None, root:str="./data", num_graphs:int=100, directed:bool=False, max_workers:int=2, type:str='erdos_renyi'):
        self.n = np.array(n)
        self.p = np.array(p) if p is not None else None
        self.m = np.array(m) if m is not None else None
        self.directed = directed
        self.path = root
        self.max_workers = max_workers
        self.num_graphs = num_graphs
        self.type = type

        # check if the directory exists
        if not os.path.exists(osp.join(self.path)):
            os.makedirs(osp.join(self.path))
        if not os.path.exists(osp.join(self.path, 'raw')):
            os.makedirs(osp.join(self.path, 'raw'))
    
    def generate(self):
        print('Generating ' + str(self.num_graphs) + ' graphs')
        with tqdm(total=self.num_graphs) as pbar:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for i in range(self.num_graphs):
                    executor.submit(self.generate_graph(i))
                    pbar.update(1)

    def generate_graph(self, i):
        # get n and p from the arrays if they are arrays or just use the values
        n = np.random.randint(self.n[0], self.n[1]) if self.n.size > 1 else int(self.n)
        m = int(np.random.randint(self.m[0], self.m[1])) if self.m is not None and self.m.size > 1 else (self.m if self.m is not None else None)
        p = np.random.uniform(self.p[0], self.p[1]) if self.p is not None and self.p.size > 1 else (self.p if self.p is not None else None)
        
        g = Graph(n=n, m=m if m is not None else None, p=p if p is not None else None, directed=self.directed, type=self.type)
        g.saveRawFile(osp.join(self.path, 'raw', 'graph_' + str(i) + '.edgelist'))

#----------------------------------------------------------
# CLASS RandomGraphDataset
while True:
    try:
        from clrs._src.algorithms.graphs import bfs
        break  # Exit the loop if import is successful
    except Exception as e:
        continue

import glob
import torch
import time

class RandomGraphDataset(Dataset):
    """A dataset of random graphs with their BFS results saved as torch tensors."""
    def __init__(self, n, p, root: str = "./data", gen_num_graph: int = 100, directed: bool = False, transform=None, pre_transform=None):
        self.n = n
        self.p = p
        self.directed = directed
        self.gen_num_graphs = gen_num_graph
        super(RandomGraphDataset, self).__init__(root, transform, pre_transform)
    
    @property
    def raw_file_names(self):
        return list(map(os.path.basename,glob.glob(osp.join(self.raw_dir, "*.edgelist"))))
    
    @property
    def processed_file_names(self):
        return list(map(os.path.basename,glob.glob(osp.join(self.processed_dir, "*.pt"))))
    
    def download(self):
        """
        Creates an instance of the graph generator and generates the graphs.

        This method initializes a graph generator instance and utilizes it to generate graphs based on the provided parameters.
        """
        print(f"Using {os.cpu_count()} CPU cores")
        gen = GraphGenerator(n=self.n, p=self.p, root=self.root, num_graphs=self.gen_num_graphs, directed=self.directed, max_workers=os.cpu_count())
        gen.generate()

#----------------------------------------------------------
# CLASS RandomGraphDataset
while True:
    try:
        from clrs._src.algorithms.graphs import bfs
        break  # Exit the loop if import is successful
    except Exception as e:
        continue

import glob
import torch

class RandomGraphDataset(Dataset):
    """A dataset of random graphs with their BFS results saved as torch tensors."""
    def __init__(self, n, m=None, p=None, root: str = "./data", gen_num_graph: int = 100, directed: bool = False, type: str = 'erdos_renyi', transform=None, pre_transform=None):
        self.n = n
        self.p = p
        self.m  = m
        self.directed = directed
        self.type = type
        self.gen_num_graphs = gen_num_graph
        super(RandomGraphDataset, self).__init__(root, transform, pre_transform)
    
    @property
    def raw_file_names(self):
        return list(map(os.path.basename,glob.glob(osp.join(self.raw_dir, "*.edgelist"))))
    
    @property
    def processed_file_names(self):
        return list(map(os.path.basename,glob.glob(osp.join(self.processed_dir, "*.pt"))))
    
    def download(self):
        """Creates an instance of the graph generator and generates the graphs."""
        gen = GraphGenerator(n=self.n, p=self.p, m=self.m, root=self.root, num_graphs=self.gen_num_graphs, directed=self.directed, type =self.type, max_workers=os.cpu_count())
        gen.generate()
           
    def process(self):
        if not os.path.exists(osp.join(self.root, 'processed')):
            os.makedirs(osp.join(self.root, 'processed'))

        with tqdm(total=self.gen_num_graphs) as pbar:
            for i in range(self.gen_num_graphs):

                g = nx.read_edgelist(osp.join(self.root, 'raw', 'graph_' + str(i) + '.edgelist'), nodetype=int)

                g = nx.from_edgelist(g.edges())

                max_node = max(g.nodes(), default=0)

                for m in range(int(max_node)+1):
                    if int(m) not in g.nodes():
                        g.add_node(str(m))

                adj = nx.to_numpy_array(g)
                node_index = [int(x) for x in g.nodes()]
                # correcting adj
                permutation = [node_index.index(i) for i in range(len(node_index))]
                adj = adj[:, permutation][permutation, :]

                edges_list_int = [(int(u), int(v)) for u, v in list(g.edges())]
                edges_list_int = edges_list_int + [(v, u) for u, v in edges_list_int]
                edge_index = torch.tensor(edges_list_int).t().contiguous()

                s = np.random.randint(0, len(adj)) # randomly choosing a starting node in the graph

                pi, probes = bfs(adj, s) # calculating the breadth first search

                pi_h = probes['hint']['node']['pi_h']['data'] # parents hints

                edges = self.get_edges(pi, edge_index) # edges that have been traversed
                edges_h = np.array([self.get_edges(x, edge_index) for x in pi_h]) # edges traversed hints

                reach_h = probes['hint']['node']['reach_h']['data']
                
                pos = np.arange(0, len(adj)) / len(adj) # position of nodes (between 0 and 1)
                length = edge_index.shape[1] # number of edges

                tmp = np.zeros(len(adj))
                tmp[s] = 1
                s = tmp

                dict_ = {'edge_index': edge_index, 'pos': pos, 'length': length, 's': s, 'pi': pi, 'reach_h': reach_h, 'pi_h': pi_h, 'edges': edges, 'edges_h' : edges_h}
                dict_ = {k: self.to_torch(v) for k, v in dict_.items()}
                dict_['hints'] = np.array(['reach_h', 'pi_h', 'edges_h'])
                dict_['inputs'] = np.array(['pos', 's'])
                dict_['outputs'] = np.array(['edges'])
                tensor = CLRSData(**dict_)

                if self.pre_transform is not None:
                    tensor = self.pre_transform(tensor)
                torch.save(tensor, osp.join(self.root, 'processed', f'data{i}.pt'))
                pbar.update(1)

    def len(self):
        """
        Returns the number of files in the dataset.

        Returns:
            int: Number of files in the dataset.
        """
        return len(self.raw_file_names)
    
    def get(self, idx:int):
        """
        Returns the idx-th graph in the dataset.

        Args:
            idx (int): Index of the graph to retrieve.

        Returns:
            torch.Tensor: The graph data.
        """
        data = torch.load(osp.join(self.root, 'processed', 'data' + str(idx) + '.pt'))
        return data

    def get_edges(self, pi, edge_index):
        """
        Generates binary edge information based on the provided node indices and edge index.

        Args:
            pi (list or numpy.ndarray): Node indices.
            edge_index (numpy.ndarray): Edge index.

        Returns:
            numpy.ndarray: Binary edge information.
        """
        n = len(pi)
        src = np.array(pi)
        des = np.arange(n)
        mask = src != des
        src = src[mask]
        des = des[mask]
        indices = self.find_edge_indices(edge_index, src, des)
        edges = np.zeros(edge_index.shape[1], dtype=np.float32)
        edges[indices] = 1
        return edges

    def find_edge_indices(self, edge_index, src, des):
        """
        Finds the indices of edges between specified source and destination nodes.

        Args:
            edge_index (numpy.ndarray): Edge index.
            src (numpy.ndarray): Source node indices.
            des (numpy.ndarray): Destination node indices.

        Returns:
            numpy.ndarray: Indices of edges between specified source and destination nodes.
        """
        src_mask = np.in1d(edge_index[0], src)
        des_mask = np.in1d(edge_index[1], des)
        mask = np.logical_and(src_mask, des_mask)
        return np.where(mask)[0]

    def to_torch(self, value):
        """
        Transforms a numpy array into a torch tensor.

        Args:
            value (numpy.ndarray or torch.Tensor or any): Input data.

        Returns:
            torch.Tensor: Transformed data as a torch tensor.
        """
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value)
        elif isinstance(value, torch.Tensor):
            return value
        else:
            return torch.tensor(value)