name=ogbn-papers100M
for n_gpu in 1 2 4 8
do
  outdir=/data/dsp
  mkdir -p $outdir
  cd $outdir
  wget https://dgl-data.s3.us-west-2.amazonaws.com/dsp-ae/${name}${n_gpu}.tar.gz || exit 1
  tar zxvf ${name}${n_gpu}.tar.gz || exit 1
  mv ${name}${n_gpu} ogb-paper100M${n_gpu}
done
