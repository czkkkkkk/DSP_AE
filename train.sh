source /root/miniconda3/etc/profile.d/conda.sh
model="graphsage"

echo dsp_epoch_time
conda activate dsp
for dataset in "products" "papers" "friendster"; do
  for rank in 1 2 4 8; do
    bash dsp/table_V.sh $dataset $rank
  done
done

conda activate dgl
echo dgl_cpu_epoch_time
for dataset in "product" "paper" "fs"; do
  for rank in 1 2 4 8; do
    bash run.sh $dataset $rank 0 0
  done
done

echo dgl_uva_epoch_time
for dataset in "product" "paper" "fs"; do
  for rank in 1 2 4 8; do
    bash run.sh $dataset $rank 1 0
  done
done

conda activate pyg
echo pyg_epoch_time
for dataset in "product" "paper" "fs"; do
  for rank in 1 2 4 8; do
    bash run.sh $dataset $rank 0
  done
done

conda activate quiver
echo quiver_epoch_time
for dataset in "product" "paper" "fs"; do
  for rank in 1 2 4 8; do
    bash run.sh $dataset $rank 0
  done
done