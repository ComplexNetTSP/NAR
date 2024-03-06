import os.path as osp
import tqdm
import glob
import os
import torch
import numpy as np
import pandas as pd
import networkit as nk
from torch_geometric.data import Dataset, Data
from sklearn.preprocessing import StandardScaler
from .gen_raw_dataset import GenerateRawDataset
 
class RandomGraphDataset(Dataset):
    def __init__(self, root: str = "./data", gen_num_graph: int = 200, num_nodes: int = 1000, p: float = 0.001, directed: bool = True, transform=None, pre_transform=None):
        self.directed = directed
        self.gen_num_graph = gen_num_graph
        self.num_nodes = num_nodes
        self.p = p
        super(RandomGraphDataset, self).__init__(
        root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        return list(map(
        os.path.basename,
        glob.glob(osp.join(self.raw_dir, "*.edgelist")))
        )
    
    @property
    def processed_file_names(self):
        return list(map(
        os.path.basename,
        glob.glob(osp.join(self.processed_dir, "*.pt")))
        )
        
    def download(self):
        gen_raw_dataset = GenerateRawDataset(
        self.root, gen_num_graph=self.gen_num_graph, num_nodes=self.num_nodes, p=self.p, isdirected=self.directed)
        gen_raw_dataset.process()
    
    def process(self):
        idx = 0
        if not os.path.exists(osp.join(self.root, 'processed')):
            os.mkdir(osp.join(self.root, 'processed'))
        for raw_path in tqdm.tqdm(self.raw_paths):
            edge_index, g = self.load_edge_csv(raw_path)
            x = self.degree_centrality(g)
            betweeness, betweeness_rank_norm, _ = self.betweeness_centrality(g)
            y = torch.cat([betweeness, betweeness_rank_norm], dim=1)
            data = Data(x=x, y=y, edge_index=edge_index, directed=self.directed,
            num_nodes=g.numberOfNodes(), num_edges=g.numberOfEdges())
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1
            
    def load_edge_csv(self, path):
        df = pd.read_csv(path, sep='\t', header=None)
        src = [index for index in df[0]]
        dst = [index for index in df[1]]
        edge_index = torch.tensor([src, dst]).int()
        g = nk.readGraph(path, nk.Format.EdgeListTabZero,
        directed=self.directed)
        return edge_index, g
        
    def len(self):
        return len(self.raw_file_names)
    
    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data
    

#_________
    
import os.path as osp
import networkx as nx
from tqdm import tqdm
import os 
 
class GenerateRawDataset:
    def __init__(self, root:str, isdirected:bool=True, gen_num_graph:int=10, num_nodes:int=1000, p:float=0.001):
        self.root = root
        self.isdirected = isdirected
        self.gen_num_graph = gen_num_graph
        self.num_nodes = num_nodes
        self.p = p
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        if not os.path.exists(osp.join(self.root, 'raw')):
            os.mkdir(osp.join(self.root, 'raw'))
    
    def save_graph(self, filename, g):
        path = osp.join(self.root, 'raw', filename)
        nx.write_edgelist(g, path, delimiter='\t', data=False)
    
    def process(self):
        barabasi_g, random_g = self.gen_num_graph, self.gen_num_graph
        print("Generate dataset...")
        with tqdm(total=barabasi_g) as pbar:
            for idx in range(barabasi_g):
                if self.isdirected:
                    g = nx.scale_free_graph(self.num_nodes)
                largest_cc = max(nx.weakly_connected_components(g), key=len)
            else:
                g = nx.barabasi_albert_graph(n=self.num_nodes, m=8) 
                largest_cc = max(nx.connected_components(g), key=len)
                gsub = nx.convert_node_labels_to_integers(g.subgraph(largest_cc).copy())
            self.save_graph(f'barabasi_isdirected_{idx}.edgelist', gsub) 
            pbar.update(1)
            
    # for idx in range(barabasi_g, barabasi_g + random_g):
    # g = nx.fast_gnp_random_graph(n=self.num_nodes, p=self.p, directed=self.isdirected)
    # self.p += 0.00002
    # if self.isdirected:
    # largest_cc = max(nx.weakly_connected_components(g), key=len)
    # else:
    # largest_cc = max(nx.connected_components(g), key=len)
    # gsub = nx.convert_node_labels_to_integers(g.subgraph(largest_cc))
    # self.save_graph(f'random_isdirected_{idx}.edgelist', gsub)
    # pbar.update(1)