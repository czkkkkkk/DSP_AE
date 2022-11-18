#!/bin/bash
dataset=$1
n_gpu=$2
datadir=/data/dsp/${dataset}${n_gpu}/${dataset}.json

export DGL_DS_USE_NCCL=1
export DGL_DS_MASTER_PORT=12210
export DGL_DS_COMM_PORT=17211

cache_ratio=50
graph_cache_ratio=100
graph_cache_gb=-1

python /root/projects/DSP_AE/dsp/sampling.py --part_config=${datadir} \
  --n_ranks=${n_gpu} \
  --graph_cache_ratio=${graph_cache_ratio} \
  --graph_cache_gb=${graph_cache_gb} \
  --sample_only
