import os
import os.path as osp

import numpy as np
import scipy.sparse as sp
import torch
from torch_sparse import coalesce
import torch_geometric
from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_zip)

class Friendster(InMemoryDataset):
  url = '/data/snap/com-friendster.ungraph.compact.txt'
  url2 = '/data/snap/com-friendster.nodetypes.txt'

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
    assert False

  def process(self):
    assert False
    node_types = []
    with open(Friendster.url2, 'r') as f:
      for line in f:
        node_type = int(line.split()[0])
        node_types.append(node_type)
    node_types = np.array(node_types)
    row = []
    col = []
    num_nodes = 0
    with open(Friendster.url, 'r') as f:
      line = f.readline()
      num_nodes = int(line)
      num_edges = 0
      cur_node = 0
      for line in f:
        for cur_edge in line.split():
          cur_edge = int(cur_edge)
          row.append(cur_node)
          col.append(cur_edge)
          num_edges += 1
        cur_node += 1
      print('number nodes:', num_nodes)
      print('number edges:', num_edges)
    row = torch.from_numpy(np.array(row)).to(torch.long)
    col = torch.from_numpy(np.array(col)).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)
    edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)
    edge_index = torch_geometric.utils.to_undirected(edge_index)
    data = Data(edge_index=edge_index)
    data.node_types = torch.from_numpy(node_types)
    torch.save(self.collate([data]), self.processed_paths[0])

if __name__ == '__main__':
  dataset = Friendster('/data/pyg/Friendster')
  print(dataset[0])