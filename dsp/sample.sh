#!/bin/bash
dataset=$1
n_gpu=$2
datadir=/data/dsp/${dataset}${n_gpu}/${dataset}.json

export DGL_DS_USE_NCCL=1
export DGL_DS_MASTER_PORT=12210
export DGL_DS_COMM_PORT=17211

echo "--------------Running DSP sampler on $dataset with $n_gpus GPUs--------------------"
if [ $dataset == "ogb-product" ]; then
  graph_cache_ratio=100
elif [ $dataset == "ogb-paper100M" ]; then
  if [ $n_gpus == "1" ]; then
    graph_cache_ratio=20
  elif [ $n_gpus == "2" ]; then
    graph_cache_ratio=50
  elif [ $n_gpus == "4" ]; then
    graph_cache_ratio=100
  elif [ $n_gpus == "8" ]; then
    graph_cache_ratio=100
  fi
elif [ $dataset == "friendster" ]; then
  if [ $n_gpus == "1" ]; then
    graph_cache_ratio=40
  elif [ $n_gpus == "2" ]; then
    graph_cache_ratio=80
  elif [ $n_gpus == "4" ]; then
    graph_cache_ratio=100
  elif [ $n_gpus == "8" ]; then
    graph_cache_ratio=100
  fi
else
  echo "Invalid dataset: $dataset"
fi

graph_cache_gb=-1

python /root/projects/DSP_AE/dsp/sampling.py --part_config=${datadir} \
  --n_ranks=${n_gpu} \
  --graph_cache_ratio=${graph_cache_ratio} \
  --graph_cache_gb=${graph_cache_gb} \
  --sample_only
