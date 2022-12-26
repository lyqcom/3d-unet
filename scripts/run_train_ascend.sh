#!/bin/bash


ulimit -m unlimited
export DEVICE_NUM=1
export DEVICE_ID=0
export RANK_SIZE=$DEVICE_NUM
export RANK_ID=0


DATA_PATH=/home/ma-user/work/data/MM-WHS-v2/train/data/
SEG_PATH=/home/ma-user/work/data/MM-WHS-v2/train/seg/
CKPT_PATH=ckpt


export PYTHONPATH=$PWD/src:$PYTHONPATH
python -u train.py  \
    --data_path=$DATA_PATH \
    --seg_path=$SEG_PATH \
    --ckpt_path=$CKPT_PATH 2>&1 | tee train_log.txt


echo 'done'

#python3 -u train.py --data_path=/home/ma-user/work/data/MM-WHS-v2/train/data/  --seg_url=/home/ma-user/work/data/MM-WHS-v2/train/seg/ --ckpt_path ./ckpt 2>&1 | tee unet3d.log
