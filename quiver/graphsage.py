# Reaches around 0.7870 ± 0.0036 test accuracy.
import os

import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torchmetrics.functional as MF
from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborSampler
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import time
from paper100M import Paper100M
from friendster import Friendster

####################
# Import Quiver
####################
import quiver

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(SAGE, self).__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all, device, subgraph_loader):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
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


def run(rank, world_size, quiver_sampler, quiver_feature, y, train_idx, num_features, num_classes, sample_only):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]

    # train_idx = train_idx.to(rank)
    # train_loader = torch.utils.data.DataLoader(train_idx, batch_size=1024)
    train_loader = torch.utils.data.DataLoader(train_idx, batch_size=1024, pin_memory=True)

    model = SAGE(num_features, 256, num_classes, num_layers=3).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fcn = nn.CrossEntropyLoss()

    y = y.to(rank)
    total_time = 0
    epoch_num = 10
    skip_epoch = 5
    time_array = []
    acc_array = []
    loss_array = []
    time_begin = time.time()
    for epoch in range(0, epoch_num):
        model.train()

        torch.cuda.synchronize()
        dist.barrier()
        tic = time.time()
        for it, seeds in enumerate(train_loader):
            n_id, batch_size, adjs = quiver_sampler.sample(seeds)
            adjs = [adj.to(rank) for adj in adjs]
            if not sample_only:
                optimizer.zero_grad()
                out = model(quiver_feature[n_id], adjs)
                y_label = y[n_id[:batch_size]]
                loss = loss_fcn(out, y_label)
                loss.backward()
                optimizer.step()
            # if it % 1 == 0:
            #     time_array.append(time.time() - time_begin)
            #     acc = MF.accuracy(out, y_label)
            #     dist.all_reduce(acc)
            #     acc = acc / world_size
            #     dist.all_reduce(loss)
            #     loss = loss / world_size
            #     acc_array.append(acc.item())
            #     loss_array.append(loss.item())
            #     if rank == 0:
            #         print('rank', rank, 'Loss', loss.item(), 'Acc', acc.item())

        torch.cuda.synchronize()
        dist.barrier()
        toc = time.time()

        if rank == 0:
            print("epoch", epoch, "time cost", toc - tic)
            if epoch >= skip_epoch:
                total_time += toc - tic

        dist.barrier()

    print("avg time:", total_time / (epoch_num - skip_epoch))
    dist.destroy_process_group()

    # if rank == 0:
    #     print("array len:", len(time_array), len(acc_array), len(loss_array))
    #     with open("quiver_product_curve.txt", 'w') as file:
    #         for i in range(len(time_array)):
    #             file.write(str(time_array[i]) + ' ')
    #             file.write(str(acc_array[i]) + ' ')
    #             file.write(str(loss_array[i]) + '\n')
    #         file.close()

if __name__ == '__main__':
    import sys
    graph_name = sys.argv[1]
    world_size = int(sys.argv[2])
    sample_only = int(sys.argv[3])
    print('Let\'s use', world_size, 'GPUs!')

    if graph_name == 'product':
      root = "/data/ogb"
      dataset = PygNodePropPredDataset('ogbn-products', root)
      data = dataset[0]
      split_idx = dataset.get_idx_split()
      train_idx, val_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
      ##############################
      # Create Sampler And Feature
      ##############################
      csr_topo = quiver.CSRTopo(data.edge_index)
      quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [15, 10, 5], 0, mode="UVA")
      feature = torch.zeros(data.x.shape)
      feature[:] = data.x
      quiver_feature = quiver.Feature(rank=0, device_list=list(range(world_size)), device_cache_size="5G", cache_policy="device_replicate", csr_topo=csr_topo)
      quiver_feature.from_cpu_tensor(feature)
      labels = data.y.squeeze()
      num_classes = dataset.num_classes
      num_features = dataset.num_features
    elif graph_name == 'paper':
      dataset = Paper100M('/data/pyg/Paper100M')
      print("finish loading paper100")
      data = dataset[0]
      num_classes = 172
      labels = data.y.type(torch.LongTensor).squeeze()
      print("finish generate fake labels")
      is_train = (dataset[0].node_types == 0)
      train_idx = torch.nonzero(is_train, as_tuple=True)[0]
      ##############################
      # Create Sampler And Feature
      ##############################
      csr_topo = quiver.CSRTopo(data.edge_index)
      quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [15, 10, 5], 0, mode="UVA")
      feature = torch.zeros(data.x.shape)
      feature[:] = data.x
      cache = "5G"
      quiver_feature = quiver.Feature(rank=0, device_list=list(range(world_size)), device_cache_size=cache, cache_policy="p2p_clique_replicate", csr_topo=csr_topo)
      quiver_feature.from_cpu_tensor(feature)
      quiver.init_p2p(list(range(world_size)))
      num_features = dataset.num_features
    elif graph_name == 'fs':
      dataset = Friendster('/data/pyg/Friendster')
      data = dataset[0]
      num_features = 256
      num_classes = 3
      num_nodes = int(dataset[0].edge_index.max()) + 1
      is_train = (dataset[0].node_types == 0)
      train_idx = torch.nonzero(is_train, as_tuple=True)[0]
      csr_topo = quiver.CSRTopo(data.edge_index)
      quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [15, 10, 5], 0, mode='UVA')
      cache = "5G"
      quiver_feature = quiver.Feature(rank=0, device_list=list(range(world_size)), device_cache_size=cache, cache_policy="p2p_clique_replicate", csr_topo=csr_topo)
      y = torch.randint(size=(num_nodes, ), low=0, high=num_classes-1).type(torch.LongTensor).squeeze()
      x = torch.rand(num_nodes, 256, dtype=torch.float32)
      quiver_feature.from_cpu_tensor(x)
      quiver.init_p2p(list(range(world_size)))

    mp.spawn(
        run,
        args=(world_size, quiver_sampler, quiver_feature, labels, train_idx, num_features, num_classes, sample_only),
        nprocs=world_size,
        join=True
    )