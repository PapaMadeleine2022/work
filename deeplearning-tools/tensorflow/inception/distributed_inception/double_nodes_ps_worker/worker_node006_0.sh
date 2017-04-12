
CUDA_VISIBLE_DEVICES=0 \
python inception_distributed_train.py \
  --num_gpus=2 \
  --batch_size=32 \
  --data_dir=/home/lichenghua/data/imagenet/tf-data/ \
  --train_dir=/home/lichenghua/data/imagenet/log_data/distributed_inception_v3/ \
  --subset=traincnns \
  --job_name="worker" \
  --task_index=0 \
  --ps_hosts="node006:2222,node007:2222" \
  --worker_hosts="node006:2223,node006:2224,node007:2223,node007:2224"