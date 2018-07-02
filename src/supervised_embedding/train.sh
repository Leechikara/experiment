#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 nohup python train.py --task task_1 --cuda > debug/train_task1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --task task_2 --cuda > debug/train_task2.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --task task_3 --cuda > debug/train_task3.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --task task_4 --cuda > debug/train_task4.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --task task_5 --cuda > debug/train_task5.log 2>&1 &