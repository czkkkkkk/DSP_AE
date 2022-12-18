#!/bin/bash
#source /efs/zkcai/miniconda3/etc/profile.d/conda.sh
#conda activate dgl
graph_name=$1
n_gpu=$2
use_uva=$3
sample_only=$4

export DGL_DS_USE_NCCL=1
export DGL_DS_MASTER_PORT=12210
export DGL_DS_COMM_PORT=17211

python /root/projects/DSP_AE/dgl/graphsage.py ${graph_name} ${n_gpu} ${use_uva} ${sample_only}
