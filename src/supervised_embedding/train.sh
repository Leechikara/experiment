#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 nohup python train.py --task task_1 --cuda > task1_debug.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --task task_2 --cuda > task2_debug.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --task task_3 --cuda > task3_debug.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --task task_4 --cuda > task4_debug.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --task task_5 --cuda > task5_debug.log 2>&1 &