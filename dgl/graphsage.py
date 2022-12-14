import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.optim
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
import time
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
import tqdm
from dgl.data import RedditDataset
from paper100M import Paper100MDataset
from friendster import FriendSterDataset
import sys
import torch.multiprocessing as mp

class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(0.5)
        self.n_hidden = n_hidden
        self.n_classes = n_classes

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size, num_workers, buffer_device=None):
        # The difference between this inference function and the one in the official
        # example is that the intermediate results can also benefit from prefetching.
        g.ndata['h'] = g.ndata['feat']
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1, prefetch_node_feats=['h'])
        dataloader = dgl.dataloading.DataLoader(
                g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
                batch_size=1000, shuffle=False, drop_last=False, num_workers=num_workers,
                persistent_workers=(num_workers > 0))
        if buffer_device is None:
            buffer_device = device

        for l, layer in enumerate(self.layers):
            y = torch.zeros(
                g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes,
                device=buffer_device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = blocks[0].srcdata['h']
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                y[output_nodes] = h.to(buffer_device)
            g.ndata['h'] = y
        return y

def train(rank, world_size, graph, num_classes, train_idx, use_uva, sample_only):
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', 'tcp://127.0.0.1:12347', world_size=world_size, rank=rank)

    model = SAGE(graph.ndata['feat'].shape[1], 256, num_classes).cuda()
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]

    if use_uva == 0:
      sampler = dgl.dataloading.NeighborSampler([15, 10, 5], replace=True)
      train_dataloader = dgl.dataloading.DataLoader(graph, train_idx, sampler,
                                                    device='cuda', batch_size=1024, shuffle=False, drop_last=False,
                                                    num_workers=int(32/world_size), use_ddp=False, use_uva=False, persistent_workers=True)
    else:
      train_idx = train_idx.to('cuda')
      if not sample_only:
        sampler = dgl.dataloading.NeighborSampler([15, 10, 5], prefetch_node_feats=['feat'], prefetch_labels=['label'], replace=True)
      else:
        sampler = dgl.dataloading.NeighborSampler([15, 10, 5], prefetch_node_feats=None, prefetch_labels=None, replace=True)
      train_dataloader = dgl.dataloading.DataLoader(graph, train_idx, sampler,
                                                    device='cuda', batch_size=1024, shuffle=False, drop_last=False,
                                                    num_workers=0, use_ddp=False, use_uva=True)

    torch.cuda.synchronize()

    durations = []
    train_duration = []
    for i in range(5):
        model.train()
        cnt = 0
        sampling_time = 0
        training_time = 0
        
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.time()

        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            if not sample_only:
                x = blocks[0].srcdata['feat']
                y = blocks[-1].dstdata['label'][:, 0]
                y_hat = model(blocks, x)
                loss = F.cross_entropy(y_hat, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
            cnt += 1
            torch.cuda.synchronize()
            dist.barrier()
        
        torch.cuda.synchronize()
        dist.barrier()
        tt = time.time()
        if rank == 0:
            print("epoch:", i, tt - t0, "batches:", cnt)
            if i >= 2:
                durations.append(tt - t0)
                train_duration.append(training_time)
        dist.barrier()
    if rank == 0:
      print("avg time cost:", np.mean(durations))
      print("avg traing time:", np.mean(train_duration))
   
if __name__ == '__main__':
    graph_name = sys.argv[1]
    n_procs = int(sys.argv[2])
    use_uva = int(sys.argv[3])
    sample_only = int(sys.argv[4])
    print("-------- Using num_gpu: {}".format(n_procs))

    if graph_name == 'product':
        dataset = DglNodePropPredDataset(name='ogbn-products', root='/data/ogb/')
        graph, labels = dataset[0]
        bigraph = dgl.to_bidirected(graph)
        bigraph.ndata['feat'] = graph.ndata['feat']
        fake_labels = torch.randint(size=labels.shape, low=0, high=dataset.num_classes-1).type(torch.LongTensor)
        bigraph.ndata['label'] = fake_labels
        bigraph.create_formats_()     # must be called before mp.spawn().
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
        num_classes = dataset.num_classes
    elif graph_name == 'paper':
        dataset = Paper100MDataset()
        bigraph = dataset[0]
        num_classes = dataset.num_classes
        num_classes = 172
        is_train = (bigraph.ndata['node_type'] == 0)
        train_idx = torch.nonzero(is_train, as_tuple=True)[0]
        bigraph.create_formats_()
    elif graph_name == 'fs':
        dataset = FriendSterDataset()
        bigraph = dataset[0]
        num_nodes = bigraph.num_nodes()
        num_classes = dataset.num_classes
        bigraph.ndata['label'] = torch.randint(size=(num_nodes, ), low=0, high=num_classes-1).type(torch.LongTensor).reshape(-1, 1)
        bigraph.ndata['feat'] = torch.rand(num_nodes, 256, dtype=torch.float32)
        is_train = (bigraph.ndata['node_type'] == 0)
        train_idx = torch.nonzero(is_train, as_tuple=True)[0]
        bigraph.edata['count'] = bigraph.edata['count'].type(torch.int8)
        bigraph.create_formats_()
        dataset = None
        is_train = None

    # Tested with mp.spawn and fork.  Both worked and got 4s per epoch with 4 GPUs
    # and 3.86s per epoch with 8 GPUs on p2.8x, compared to 5.2s from official examples.
    mp.spawn(train, args=(n_procs, bigraph, num_classes, train_idx, use_uva, sample_only), nprocs=n_procs)
