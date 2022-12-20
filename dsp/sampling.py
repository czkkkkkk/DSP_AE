from torch.nn.parallel import DistributedDataParallel
from model import SAGE
import dgl.ds as ds
import time
import torch.multiprocessing as mp
import torch.distributed as dist
import dgl
import torch as th
import argparse
from dgl.data import register_data_args
import os

from dgl.ds.utils import *
from dgl.ds.pc_queue import *

os.environ["DGLBACKEND"] = "pytorch"


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12377"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def run(rank, args):
    print("Start rank", rank, "with args:", args)
    th.cuda.set_device(rank)
    setup(rank, args.n_ranks)
    num_epochs = 10
    batch_size = 1024

    ds.init(
        rank,
        args.n_ranks,
        enable_comm_control=False,
        enable_profiler=False,
        enable_kernel_control=False,
    )
    data = load_partition(args.part_config, rank, batch_size, args)

    fanout = [5, 10, 15]
    sampler = NeighborSampler(
        data.train_g,
        data.num_vertices,
        data.min_vids,
        data.min_eids,
        data.global_nid_map,
        fanout,
        dgl.ds.sample_neighbors,
        data.device,
    )

    dataloader = dgl.dataloading.NodeDataLoader(
        data.train_g,
        data.train_nid,
        sampler,
        device=data.device,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    s = th.cuda.Stream(device=data.device)
    dgl.ds.set_device_thread_local_stream(data.device, s)

    th.distributed.barrier()
    total = 0
    skip_epoch = 5

    for epoch in range(num_epochs):
        tic = time.time()
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            pass
        th.cuda.synchronize()
        th.distributed.barrier()
        toc = time.time()
        if rank == 0:
            print('Rank: ', rank, 'world_size: ', args.n_ranks, 'sampling time', toc - tic)
        if epoch >= skip_epoch:
            total += (toc - tic)

    if rank == 0:
        print("rank:", rank, 'world_size: ', args.n_ranks, "sampling time:", total/(num_epochs - skip_epoch))
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN")
    register_data_args(parser)
    parser.add_argument("--graph_name", default="test", type=str, help="graph name")
    parser.add_argument(
        "--part_config",
        default="/data/ds/metis_ogbn-papers100M4/ogbn-papers100M.json",
        type=str,
        help="The path to the partition config file",
    )
    parser.add_argument("--sample_only", action='store_true')
    parser.add_argument("--n_ranks", default=4, type=int, help="Number of ranks")
    parser.add_argument(
        "--graph_cache_ratio",
        default=100,
        type=int,
        help="Ratio of edges cached in the GPU",
    )
    parser.add_argument(
        "--graph_cache_gb",
        default=-1,
        type=int,
        help="Memory used to cache graph topology. Setting it not equal to -1 disables graph_cache_ratio",
    )
    parser.add_argument(
        "--in_feats",
        default=256,
        type=int,
        help="In feature dimension used when the graph do not have feature",
    )
    args = parser.parse_args()

    mp.spawn(run, args=(args,), nprocs=args.n_ranks, join=True)
