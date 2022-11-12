from torch.nn.parallel import DistributedDataParallel
from model import SAGE
import dgl.ds as ds
import time
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn as nn
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
    sampler_number = 1
    loader_number = 1
    num_epochs = 10
    batch_size = 1024

    ds.init(
        rank,
        args.n_ranks,
        thread_num=sampler_number + loader_number,
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
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    s = th.cuda.Stream(device=data.device)
    dgl.ds.set_device_thread_local_stream(data.device, s)

    # Define model and optimizer
    with th.cuda.stream(s):
        model = SAGE(data.in_feats, 256, data.n_classes, len(fanout), th.relu, 0.5)
        model = model.to(data.device)
        if args.n_ranks > 1:
            model = DistributedDataParallel(
                model, device_ids=[rank], output_device=rank
            )
        loss_fcn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.003)

    th.distributed.barrier()
    total = 0
    skip_epoch = 5

    my_dataloader = ParallelNodeDataLoader(dataloader)

    total_batches = dataloader.__len__()
    batch_per_sampler = divide(total_batches, sampler_number)

    sample_worker = Sampler(
        my_dataloader,
        rank,
        num_epochs,
        0,
        sampler_number,
        batch_per_sampler,
        loader_number,
    )
    sample_worker.start()

    batch_per_loader = divide(total_batches, loader_number)

    load_worker = SubtensorLoader(
        data.train_label,
        data.min_vids,
        sample_worker.mpmc_queue,
        rank,
        num_epochs,
        data.in_feats,
        0,
        sampler_number,
        loader_number,
        batch_per_loader,
    )
    load_worker.start()

    train_time = 0
    for epoch in range(num_epochs):
        tic = time.time()
        step = 0
        for i in range(total_batches):
            batch_data = load_worker.out_pc_queue.get(0)
            begin = time.time()
            with th.cuda.stream(s):
                batch_inputs = batch_data[0]
                batch_labels = batch_data[1]
                blocks = batch_data[2]
                batch_pred = model(blocks, batch_inputs)
                loss = loss_fcn(batch_pred, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                s.synchronize()
                th.distributed.barrier()
            step += time.time() - begin

        toc = time.time()
        if epoch >= skip_epoch:
            total += toc - tic
            train_time += step
        if rank == 0:
            print(
                "Epoch {}, rank {}, train time {}, epoch time {}".format(
                    epoch, rank, step, toc - tic
                )
            )
    if rank == 0:
        print(
            "rank: {} average epoch time: {}".format(
                rank, total / (num_epochs - skip_epoch)
            )
        )
    sample_worker.join()
    load_worker.join()
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
    parser.add_argument("--n_ranks", default=4, type=int, help="Number of ranks")
    parser.add_argument(
        "--feat_mode",
        default="PartitionCache",
        type=str,
        help="Feature cache mode. (AllCache, PartitionCache, ReplicateCache)",
    )
    parser.add_argument(
        "--cache_ratio", default=10, type=int, help="Percentages of features on GPUs"
    )
    parser.add_argument(
        "--graph_cache_ratio",
        default=100,
        type=int,
        help="Ratio of edges cached in the GPU",
    )
    parser.add_argument(
        "--in_feats",
        default=256,
        type=int,
        help="In feature dimension used when the graph do not have feature",
    )
    parser.add_argument(
        "--graph_cache_gb",
        default=-1,
        type=int,
        help="Memory used to cache graph topology. Setting it not equal to -1 disables graph_cache_ratio",
    )
    parser.add_argument(
        "--feat_cache_gb",
        default=-1,
        type=int,
        help="Memory used to cache features, Setting it not equal to -1 disables cache_ratio",
    )
    args = parser.parse_args()

    mp.spawn(run, args=(args,), nprocs=args.n_ranks, join=True)
