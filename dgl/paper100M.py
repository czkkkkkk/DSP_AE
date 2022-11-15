from __future__ import absolute_import

import scipy.sparse as sp
from scipy.sparse import coo_matrix
import numpy as np
import os

from dgl.data.dgl_dataset import DGLDataset, DGLBuiltinDataset
from dgl.data.utils import _get_dgl_url, generate_mask_tensor, load_graphs, save_graphs, deprecate_property
import dgl.backend as F
from dgl.convert import from_scipy
from dgl.transforms import reorder_graph
from ogb.nodeproppred import DglNodePropPredDataset

class Paper100MDataset(DGLDataset):
  raw_dir = '/data/dgl/'

  def __init__(self, raw_dir=None, force_reload=False,
               verbose=False, transform=None):
    self.num_classes = 172
    super(Paper100MDataset, self).__init__(name='paper100M',
                                           url=None,
                                           raw_dir=Paper100MDataset.raw_dir,
                                           force_reload=force_reload,
                                           verbose=verbose,
                                           transform=transform)

  def process(self):
    dataset = DglNodePropPredDataset(name='ogbn-papers100M', root='/data/ogb/')
    graph, labels = dataset[0]
    # print(graph)
    # print(labels, labels.shape)
    # exit(0)
    graph = dgl.to_bidirected(graph)
    graph.ndata['feat'] = dataset[0][0].ndata['feat']
    # fake_labels = np.random.randint(0, self.num_classes, size=labels.shape[0])
    graph.ndata['label'] = F.tensor(labels, dtype=F.data_type_dict['int64'])
    node_types = np.ones(labels.shape[0], dtype=int)
    node_types[:] = 4
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
    node_types[train_idx] = 0
    node_types[valid_idx] = 1
    node_types[test_idx] = 2
    graph.ndata['node_type'] = F.tensor(node_types, dtype=F.data_type_dict['int32'])
    self._graph = graph

  def has_cache(self):
    graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
    print("check cache", graph_path)
    if os.path.exists(graph_path):
      print("using cached data")
      return True
    return False
  
  def save(self):
    graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
    save_graphs(graph_path, self._graph)

  def load(self):
    print("loading graph")
    graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
    graphs, _ = load_graphs(graph_path)
    print("finish loading graph")
    self._graph = graphs[0]

  def __getitem__(self, idx):
    assert idx == 0, "Reddit Dataset only has one graph"
    return self._graph

  def __len__(self):
    return 1

import dgl
if __name__ == '__main__':
  dataset = Paper100MDataset()
  graph = dataset[0]
  print(graph)
  print(graph.ndata)
  #dataset = DglNodePropPredDataset(name='ogbn-products', root='/data/dgl/')