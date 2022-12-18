source /root/miniconda3/etc/profile.d/conda.sh
model="graphsage"

conda activate dgl
echo dgl_cpu_epoch_time
mkdir -p logs/dgl-cpu/train
for dataset in "product" "paper" "fs"; do
  for rank in 1 2 4 8; do
    bash dgl/run.sh $dataset $rank 0 0 >& logs/dgl-cpu/train/${dataset}_${rank}gpus.txt
    sleep 30
  done
done

echo dgl_uva_epoch_time
mkdir -p logs/dgl-uva/train
for dataset in "product" "paper" "fs"; do
  for rank in 1 2 4 8; do
    bash dgl/run.sh $dataset $rank 1 0 >& logs/dgl-uva/train/${dataset}_${rank}gpus.txt
    sleep 30
  done
done

conda activate quiver
echo pyg_epoch_time
mkdir -p logs/pyg/train
for dataset in "product" "paper" "fs"; do
  for rank in 1 2 4 8; do
    bash pyg/run.sh $dataset $rank 0 >& logs/pyg/train/${dataset}_${rank}gpus.txt
    sleep 30
  done
done

echo quiver_epoch_time
mkdir -p logs/quiver/train
for dataset in "product" "paper" "fs"; do
  for rank in 1 2 4 8; do
    bash quiver/run.sh $dataset $rank 0 >& logs/quiver/train/${dataset}_${rank}gpus.txt
    sleep 30
  done
done

echo dsp_epoch_time
conda activate dsp
mkdir -p logs/dsp/train
for dataset in "ogb-product" "ogb-paper100M" "friendster"; do
  for rank in 1 2 4 8; do
    bash dsp/train.sh $dataset $rank >& logs/dsp/train/${dataset}_${rank}gpus.txt
    rm -rf /dev/shm/*
    sleep 30
  done
done


