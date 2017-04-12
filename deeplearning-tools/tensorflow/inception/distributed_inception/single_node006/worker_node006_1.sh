
CUDA_VISIBLE_DEVICES=1 \
python inception_distributed_train.py \
  --num_gpus=8 \
  --batch_size=32 \
  --data_dir=/home/lichenghua/data/imagenet/tf-data/ \
  --train_dir=/home/lichenghua/data/imagenet/log_data/distributed_inception_v3/ \
  --subset=traincnns \
  --job_name="worker" \
  --task_index=1 \
  --ps_hosts="node006:2222" \
  --worker_hosts="node006:2223,node006:2224,node006:2225,node006:2226,node006:2227,node006:2228,node006:2229,node006:2230"