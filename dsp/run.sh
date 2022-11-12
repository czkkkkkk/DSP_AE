#!/bin/bash
#source /efs/zkcai/miniconda3/etc/profile.d/conda.sh
#conda activate dsdgl
which python
n_gpu=$1
datadir=./data/reddit${n_gpu}/reddit.json
# datadir=/data/ds/metis_ogbn-papers100M${n_gpu}/ogbn-papers100M.json
datadir=/data/ds/distdgl/ogbn-papers100M${n_gpu}/ogb-paper100M.json
datadir=/data/ds/ogbn-papers100M${n_gpu}/ogb-paper100M.json
datadir=/efs/zkcai/projects/dsdgl/examples/pytorch/graphsage/ds/data/ogb-product${n_gpu}/ogb-product.json

fanout=5,10,15

export DGL_DS_USE_NCCL=1
export DGL_DS_MASTER_PORT=12210
export DGL_DS_COMM_PORT=17211

cache_ratio=50
graph_cache_ratio=100
feat_cache_gb=-1
graph_cache_gb=-1
feat_mode=AllCache

python graphsage.py --part_config=${datadir} \
  --n_ranks=${n_gpu} \
  --cache_ratio=${cache_ratio} \
  --feat_mode=${feat_mode} \
  --graph_cache_ratio=${graph_cache_ratio} \
  --graph_cache_gb=${graph_cache_gb} \
  --feat_cache_gb=${feat_cache_gb}
