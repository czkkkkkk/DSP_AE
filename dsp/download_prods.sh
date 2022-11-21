for n_gpu in 1
do
  outdir=/data/dsp
  mkdir -p $outdir
  aws s3 cp s3://dgl-data/dsp-ae/ogb-product${n_gpu}.tar.gz $outdir  --no-sign-request --recursive
  cd /data/dsp
  tar zxvf ogb-product${n_gpu}.tar.gz
done
