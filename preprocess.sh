#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
wget http://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz
gzip -d com-friendster.ungraph.txt.gz
mv com-friendster.ungraph.txt /data/
#compact friendster
python check.py
#start preprocess dgl data
conda activate dgl
python dgl/friendster.py
python dgl/paper100M.py
#start preprocess pyg data
conda activate pyg
python pyg/friendster.py
python pyg/paper100M.py
conda activate base
sleep 10