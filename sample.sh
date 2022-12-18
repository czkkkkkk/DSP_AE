source /root/miniconda3/etc/profile.d/conda.sh
model="graphsage"

conda activate dgl
echo dgl_cpu_epoch_time
mkdir -p logs/dgl-cpu/sample
for dataset in "product" "paper" "fs"; do
  for rank in 1 2 4 8; do
    bash dgl/run.sh $dataset $rank 0 1 >& logs/dgl-cpu/sample/${dataset}_${rank}gpus.txt
  done
done

echo dgl_uva_epoch_time
mkdir -p logs/dgl-uva/sample
for dataset in "product" "paper" "fs"; do
  for rank in 1 2 4 8; do
    bash dgl/run.sh $dataset $rank 1 1 >& logs/dgl-uva/sample/${dataset}_${rank}gpus.txt
  done
done

conda activate quiver
echo pyg_epoch_time
mkdir -p logs/pyg/sample
for dataset in "product" "paper" "fs"; do
  for rank in 1 2 4 8; do
    bash pyg/run.sh $dataset $rank 1 >& logs/pyg/sample/${dataset}_${rank}gpus.txt
  done
done

conda activate quiver
echo quiver_epoch_time
mkdir -p logs/quiver/sample
for dataset in "product" "paper" "fs"; do
  for rank in 1 2 4 8; do
    bash quiver/run.sh $dataset $rank 1 >&  logs/quiver/sample/${dataset}_${rank}gpus.txt
  done
done

echo dsp_epoch_time
conda activate dsp
mkdir -p logs/dsp/sample
for dataset in "ogb-product" "ogb-paper100M" "friendster"; do
  for rank in 1 2 4 8; do
    bash dsp/sample.sh $dataset $rank >& logs/dsp/sample/${dataset}_${rank}gpus.txt
    rm -rf /dev/shm/*
  done
done

