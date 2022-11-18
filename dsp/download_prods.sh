for n_gpu in 1 2 4 8
do
  outdir=/data/dsp/ogb-product${n_gpu}/
  mkdir -p $outdir
  aws s3 cp s3://dsp-ae/dsp/ogb-product${n_gpu}/ $outdir  --no-sign-request --recursive
done
