
CUDA_VISIBLE_DEVICES='' \
python inception_distributed_train.py \
  --num_gpus=8 \
  --batch_size=32 \
  --data_dir=/home/lichenghua/data/imagenet/tf-data/ \
  --train_dir=/home/lichenghua/data/imagenet/log_data/distributed_inception_v3/ \
  --subset=traincnns \
  --job_name='ps' \
  --task_index=0 \
  --ps_hosts='node005:2222' \
  --worker_hosts='node005:2223,node005:2224'
