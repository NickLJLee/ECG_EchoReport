export OMP_NUM_THREADS=4

wandb online
cd /data1/1shared/lijun/ecg/ECG-EchoReport/pretrain
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=101 --rdzv_endpoint=localhost:29502 main.py
