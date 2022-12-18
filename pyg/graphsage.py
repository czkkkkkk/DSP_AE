import os

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv
import torch_geometric as pyg
import time
from torch_sparse import SparseTensor
from torch_geometric.nn import ChebConv, GCNConv  # noqa
from paper100M import Paper100M
from friendster import Friendster

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2):
        super().__init__()
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(self.num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    @torch.no_grad()
    def inference(self, x_all, device, subgraph_loader):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all

def run(rank, world_size, x, adj, num_features, num_classes, train_idx, labels, sample_only):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    # data = dataset[0]
    # train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    # train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]
    # split_idx = dataset.get_idx_split()
    # train_idx, val_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
    train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]

    sizes=[15, 10, 5]
    train_loader = NeighborSampler(adj, node_idx=train_idx,
                                   sizes=sizes, batch_size=1024,
                                   shuffle=True, persistent_workers=True,
                                   num_workers=int(32/world_size), return_e_id=False)

    torch.manual_seed(12345)
    model = SAGE(num_features, 256, num_classes, num_layers=len(sizes)).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # x, y = data.x.to(rank), data.y.squeeze().to(rank)
    # y = labels.squeeze()
    # x = x.to(rank)
    y = labels.squeeze().to(rank)
    total_time = 0
    epoch_num = 10
    skip_epoch = 5

    for epoch in range(0, epoch_num):
        model.train()
        torch.cuda.synchronize()
        dist.barrier()
        tic = time.time()

        for batch_size, n_id, adjs in train_loader:
            adjs = [adj.to(rank) for adj in adjs]
            if not sample_only:
                optimizer.zero_grad()
                out = model(x[n_id].to(rank), adjs)
                loss = F.nll_loss(out, y[n_id[:batch_size]])
                loss.backward()
                optimizer.step()
            torch.cuda.synchronize()
            dist.barrier()

        torch.cuda.synchronize()
        dist.barrier()
        toc = time.time()

        if rank == 0:
            print("time cost:", toc - tic)
            if epoch >= skip_epoch:
                total_time += toc - tic

        dist.barrier()
    print("avg time cost:", total_time / (epoch_num - skip_epoch))
    dist.destroy_process_group()


if __name__ == '__main__':
    import sys
    graph_name = sys.argv[1]
    world_size = int(sys.argv[2])
    sample_only = int(sys.argv[3])
    print('Let\'s use', world_size, 'GPUs!')

    if graph_name == 'product':
      root = "/data/ogb"
      dataset = PygNodePropPredDataset('ogbn-products', root)
      split_idx = dataset.get_idx_split()
      train_idx, val_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
      num_classes = dataset.num_classes
      num_features = dataset.num_features
      x = dataset[0].x
      fake_labels = dataset[0].y.squeeze()

      num_nodes = int(dataset[0].edge_index.max()) + 1
      adj = SparseTensor(row=dataset[0].edge_index[0], col=dataset[0].edge_index[1],
                                        value=None,
                                        sparse_sizes=(num_nodes, num_nodes)).t()
    elif graph_name == 'paper':
      dataset = Paper100M('/data/pyg/Paper100M')
      num_classes = 172
      fake_labels = torch.randint(size=dataset[0].y.shape, low=0, high=num_classes-1).type(torch.LongTensor)
      is_train = (dataset[0].node_types == 0)
      train_idx = torch.nonzero(is_train, as_tuple=True)[0]
      num_features = dataset.num_features
      x = dataset[0].x
      data = dataset[0]
      num_nodes = int(data.edge_index.max()) + 1
      adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1],
                                      value=None,
                                      sparse_sizes=(num_nodes, num_nodes)).t()

      value = torch.arange(adj.nnz())
      adj = adj.set_value(value, layout='coo')
      dataset = None
      data = None
    elif graph_name == 'fs':
      dataset = Friendster('/data/pyg/Friendster')
      num_features = 256
      num_classes = 3
      num_nodes = int(dataset[0].edge_index.max()) + 1
      is_train = (dataset[0].node_types == 0)
      train_idx = torch.nonzero(is_train, as_tuple=True)[0]
      adj = SparseTensor(row=dataset[0].edge_index[0], col=dataset[0].edge_index[1],
                                        value=None,
                                        sparse_sizes=(num_nodes, num_nodes)).t()
      num_edges = adj.nnz()
      value = torch.arange(num_edges)
      adj = adj.set_value(value, layout='coo')
      fake_labels = torch.randint(size=(num_nodes, ), low=0, high=num_classes-1).type(torch.LongTensor)
      x = torch.rand(num_nodes, 256, dtype=torch.float32)
      dataset = None      

    mp.spawn(run, args=(world_size, x, adj, num_features, num_classes, train_idx, fake_labels, sample_only), nprocs=world_size, join=True)
