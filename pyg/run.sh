#!/bin/bash
#source /efs/zkcai/miniconda3/etc/profile.d/conda.sh
#conda activate pyg
graph_name=$1
n_gpu=$2
sample_only=$3

export DGL_DS_USE_NCCL=1
export DGL_DS_MASTER_PORT=12210
export DGL_DS_COMM_PORT=17211

python /root/projects/DSP_AE/pyg/graphsage.py ${graph_name} ${n_gpu} ${sample_only}
