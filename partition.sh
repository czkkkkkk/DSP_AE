#!/bin/bash

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 dataset n_parts"
  exit 0
fi

dataset=$1
n_parts=$2

if [ $dataset != "products" ] && [ $dataset != "papers" ] && [ $dataset != "friendster" ]; then
  echo "Dataset should be one of [products, papaers, friendster]"
fi

if [ $dataset == "products"]; then
  $dataset=ogb-product
elif [ $dataset == "papers"]; then
  $dataset=ogbn-papers100M
fi

outdir=/data/ds/$dataset-data-$n_gpus
ogb_root=/data/ogb

source /root/miniconda3/etc/profile.d/conda.sh
conda activate dsp

dir=/root/dsdgl/examples/pytorch/graphsage/ds
script=partition_graph.py

python $dir/$script --dataset=${dataset} --num_parts=${n_parts} --output=$outdir --root=${ogb_root}