#!/bin/bash
dataset=$1
n_gpus=$2
datadir=/data/ds/distdgl/${dataset}${n_gpus}/${dataset}.json

export DGL_DS_USE_NCCL=1
export DGL_DS_MASTER_PORT=12210
export DGL_DS_COMM_PORT=17211

#feat_mode=PartitionCache

echo "--------------Running DSP on $dataset with $n_gpu GPUs--------------------"
if [[ $dataset == "ogb-product" ]]; then
  feat_mode=AllCache
  feat_cache_ratio=100
  graph_cache_ratio=100
elif [[ $dataset == "ogb-paper100M" ]]; then
  feat_mode=PartitionCache
  if [[ $n_gpus == "1" ]]; then
    feat_cache_ratio=5
    graph_cache_ratio=20
  elif [[ $n_gpus == "2" ]]; then
    feat_cache_ratio=10
    graph_cache_ratio=20
  elif [[ $n_gpus == "4" ]]; then
    feat_cache_ratio=20
    graph_cache_ratio=20
  elif [[ $n_gpus == "8" ]]; then
    feat_cache_ratio=50
    graph_cache_ratio=100
  fi
elif [[ $dataset == "friendster" ]]; then
  feat_mode=PartitionCache
  if [[ $n_gpus == "1" ]]; then
    feat_cache_ratio=10
    graph_cache_ratio=0
  elif [[ $n_gpus == "2" ]]; then
    feat_cache_ratio=20
    graph_cache_ratio=0
  elif [[ $n_gpus == "4" ]]; then
    feat_cache_ratio=40
    graph_cache_ratio=0
  elif [[ $n_gpus == "8" ]]; then
    feat_cache_ratio=50
    graph_cache_ratio=20
  fi
else
  echo "Invalid dataset: $dataset"
fi
graph_cache_gb=-1
feat_cache_gb=-1


python /root/projects/DSP_AE/dsp/graphsage.py --part_config=${datadir} \
  --n_ranks=${n_gpus} \
  --cache_ratio=${feat_cache_ratio} \
  --feat_mode=${feat_mode} \
  --graph_cache_ratio=${graph_cache_ratio} \
  --graph_cache_gb=${graph_cache_gb} \
  --feat_cache_gb=${feat_cache_gb}
