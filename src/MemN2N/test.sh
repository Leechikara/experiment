#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 nohup python test.py --trained_task task_1 --testing_task task_1 --trained_model checkpoints/task_1/epoch_58_accuracy_0.9994848193433163.pkl --cuda > debug/test_task1_based_task1.log 2>&1 &
