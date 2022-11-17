#!/bin/bash
n_gpu=$1
datadir=/efs/zkcai/projects/dsdgl/examples/pytorch/graphsage/ds/data/ogb-product${n_gpu}/ogb-product.json

export DGL_DS_USE_NCCL=1
export DGL_DS_MASTER_PORT=12210
export DGL_DS_COMM_PORT=17211

cache_ratio=50
graph_cache_ratio=100
graph_cache_gb=-1

python sampling.py --part_config=${datadir} \
  --n_ranks=${n_gpu} \
  --graph_cache_ratio=${graph_cache_ratio} \
  --graph_cache_gb=${graph_cache_gb} \
  --sample_only
