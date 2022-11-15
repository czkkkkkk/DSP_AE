import os
import os.path as osp

import numpy as np
import scipy.sparse as sp
import torch
from torch_sparse import coalesce
import torch_geometric
from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_zip)
from ogb.nodeproppred import PygNodePropPredDataset

class Paper100M(InMemoryDataset):
  def __init__(self, root, transform=None, pre_transform=None):
    super().__init__(root, transform, pre_transform)
    self.data, self.slices = torch.load(self.processed_paths[0])
  
  @property
  def raw_file_names(self):
    return 'empty.txt'
  
  @property
  def processed_file_names(self):
    return 'data.pt'
  
  def download(self):
    pass

  def process(self):
    root = "/data/ogb/"
    dataset = PygNodePropPredDataset('ogbn-papers100M', root)
    print(dataset)
    graph, labels = dataset[0]
    print(labels, labels.shape)
    exit(0)
    edge_index = torch_geometric.utils.to_undirected(dataset[0].edge_index)
    x = dataset[0].x
    y = dataset[0].y
    data = Data(x=x, edge_index=edge_index, y=y)
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
    node_types = np.ones(y.shape[0], dtype=int)
    node_types[:] = 4
    node_types[train_idx] = 0
    node_types[valid_idx] = 1
    node_types[test_idx] = 2
    data.node_types = torch.from_numpy(node_types)
    torch.save(self.collate([data]), self.processed_paths[0])

if __name__ == '__main__':
  dataset = Paper100M('/data/pyg/Paper100M')
  print(dataset)
  root = "/data/pyg"
  dataset = PygNodePropPredDataset('ogbn-products', root)