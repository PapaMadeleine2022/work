import os
import numpy as np
#import shutil 

train_dir = '/home/lichenghua/imagenet/distributed_inception'
script_dir = ''
#if os.path.exists(os.path.join(train_dir, script_dir)):
#  shutil.rmtree(os.path.join(train_dir, script_dir))
#os.mkdir(os.path.join(train_dir, script_dir))

worker_node_list = "node006,node007"
worker_gpu_list = "0,1,2,3,4,5,6,7;0,1,2,3,4,5,6,7"

ps_node_list = worker_node_list

def generate_ps(script_name, ps_line, wh_line, task_index):
  f = open(script_name, 'w')
  f.write('\n')
  f.write('CUDA_VISIBLE_DEVICES=\'\' \\\n')
  f.write('python inception_distributed_train.py \\\n')
  f.write('  --num_gpus=8 \\\n')
  f.write('  --batch_size=32 \\\n')
  f.write('  --data_dir=/home/lichenghua/data/imagenet/tf-data/ \\\n')
  f.write('  --train_dir=/home/lichenghua/data/imagenet/log_data/distributed_inception_v3/ \\\n')
  f.write('  --subset=traincnns \\\n')
  f.write('  --job_name="ps" \\\n')
  f.write('  --task_index='+task_index+' \\\n')
  f.write(ps_line)
  f.write(wh_line)

def generate_worker(script_name, ps_line, wh_line, worker_node, gpu, task_index):
  f = open(script_name, 'w')
  f.write('\n')
  f.write('CUDA_VISIBLE_DEVICES='+gpu+' \\\n')
  f.write('python inception_distributed_train.py \\\n')
  f.write('  --num_gpus=8 \\\n')
  f.write('  --batch_size=32 \\\n')
  f.write('  --data_dir=/home/lichenghua/data/imagenet/tf-data/ \\\n')
  f.write('  --train_dir=/home/lichenghua/data/imagenet/log_data/distributed_inception_v3/ \\\n')
  f.write('  --subset=traincnns \\\n')
  f.write('  --job_name="worker" \\\n')
  f.write('  --task_index='+task_index+' \\\n')
  f.write(ps_line)
  f.write(wh_line)


if __name__=='__main__': 
  # Parse the nodes & gpus
  worker_node_list = worker_node_list.split(',')
  ps_node_list = ps_node_list.split(',')
  
  worker_gpu_list = worker_gpu_list.split(';')

  # get the 'worker_hosts' line 
  wh_line = ''
  for i in range(len(worker_node_list)):
    worker_node = worker_node_list[i]
    gpu_list = worker_gpu_list[i].split(',')
    for j in range(len(gpu_list)):
      gpu = gpu_list[j]
      wh_line += worker_node+':'+str(2223+j)+','
      print('>>%s' % (wh_line))
  wh_line = '  --worker_hosts="'+wh_line[:-1]+'"'
  print('>>')
  print(' %s' % (wh_line))
  print('>>')

  # get the 'ps_hosts' line @multiple 'ps'
  ps_line = ''
  for i in range(len(ps_node_list)):
    ps_node = ps_node_list[i]
    ps_line += ps_node+':2222,'
    print('>>%s' % (ps_line))
  ps_line = '  --ps_hosts="'+ps_line[:-1]+'" \\\n'
  print('>>')
  print(' %s' % (ps_line))
  print('>>')

  run_ps = []
  # generate each 'ps' script  @multiple 'ps'
  for i in range(len(ps_node_list)):
    script_name = os.path.join(script_dir,'ps_'+ps_node_list[i]+'.sh')
    print('>>ps : %s' % script_name)
    generate_ps(script_name, ps_line, wh_line, str(i))
    run_ps.append(script_name)
  
  run_worker = []
  # generate each 'worker' script 
  task_index = -1
  for i in range(len(worker_node_list)):
    worker_node=worker_node_list[i]
    gpu_list = worker_gpu_list[i].split(',')

    run_worker_node = []
    for j in range(len(gpu_list)):
      task_index += 1
      gpu = gpu_list[j]
      script_name = os.path.join(script_dir,'worker_'+worker_node+'_'+gpu+'.sh')
      print('>>ps : %s' % script_name)
      generate_worker(script_name, ps_line, wh_line, worker_node, gpu, str(task_index))
      run_worker_node.append(script_name)
   
    run_worker.append(run_worker_node)
      
  print run_ps
  print ''
  print run_worker

"""
  # generate 'run_file' for each node
  assert(len(ps_node_list)==len(worker_node_list))
  for i in range(len(ps_node_list)):
    run_line = '' 
    node = ps_node_list[i]
    run_file_name = os.path.join(train_dir, 'run_'+node+'.sh')
    print('<< %s' % run_file_name)
    if i==0:
      run_line += 'sh '+run_ps[i]+' & ' 

    run_gpu = run_worker[i]
    for j in range(len(run_gpu)):
      run_line += 'sh '+run_gpu[j]+' & ' 

    # write
    f = open(run_file_name, 'w') 
    run_line = run_line[:-3]
    f.write(run_line)
    print('++++')
    print('%s' % run_line)
    print('++++')
"""
