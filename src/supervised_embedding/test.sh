#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 nohup python test.py --trained_task task_1 --testing_task task_1 --trained_model checkpoints/task_1 --cuda > debug/test_task1_based_task1.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python test.py --trained_task task_1 --testing_task task_2 --trained_model checkpoints/task_1 --cuda > debug/test_task2_based_task1.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python test.py --trained_task task_1 --testing_task task_3 --trained_model checkpoints/task_1 --cuda > debug/test_task3_based_task1.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python test.py --trained_task task_1 --testing_task task_4 --trained_model checkpoints/task_1 --cuda > debug/test_task4_based_task1.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python test.py --trained_task task_1 --testing_task task_5 --trained_model checkpoints/task_1 --cuda > debug/test_task5_based_task1.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python test.py --trained_task task_2 --testing_task task_1 --trained_model checkpoints/task_2 --cuda > debug/test_task1_based_task2.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python test.py --trained_task task_2 --testing_task task_2 --trained_model checkpoints/task_2 --cuda > debug/test_task2_based_task2.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python test.py --trained_task task_2 --testing_task task_3 --trained_model checkpoints/task_2 --cuda > debug/test_task3_based_task2.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python test.py --trained_task task_2 --testing_task task_4 --trained_model checkpoints/task_2 --cuda > debug/test_task4_based_task2.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python test.py --trained_task task_2 --testing_task task_5 --trained_model checkpoints/task_2 --cuda > debug/test_task5_based_task2.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python test.py --trained_task task_3 --testing_task task_1 --trained_model checkpoints/task_3 --cuda > debug/test_task1_based_task3.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python test.py --trained_task task_3 --testing_task task_2 --trained_model checkpoints/task_3 --cuda > debug/test_task2_based_task3.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python test.py --trained_task task_3 --testing_task task_3 --trained_model checkpoints/task_3 --cuda > debug/test_task3_based_task3.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python test.py --trained_task task_3 --testing_task task_4 --trained_model checkpoints/task_3 --cuda > debug/test_task4_based_task3.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python test.py --trained_task task_3 --testing_task task_5 --trained_model checkpoints/task_3 --cuda > debug/test_task5_based_task3.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python test.py --trained_task task_4 --testing_task task_1 --trained_model checkpoints/task_4 --cuda > debug/test_task1_based_task4.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python test.py --trained_task task_4 --testing_task task_2 --trained_model checkpoints/task_4 --cuda > debug/test_task2_based_task4.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python test.py --trained_task task_4 --testing_task task_3 --trained_model checkpoints/task_4 --cuda > debug/test_task3_based_task4.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python test.py --trained_task task_4 --testing_task task_4 --trained_model checkpoints/task_4 --cuda > debug/test_task4_based_task4.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python test.py --trained_task task_4 --testing_task task_5 --trained_model checkpoints/task_4 --cuda > debug/test_task5_based_task4.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python test.py --trained_task task_5 --testing_task task_1 --trained_model checkpoints/task_5 --cuda > debug/test_task1_based_task5.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python test.py --trained_task task_5 --testing_task task_2 --trained_model checkpoints/task_5 --cuda > debug/test_task2_based_task5.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python test.py --trained_task task_5 --testing_task task_3 --trained_model checkpoints/task_5 --cuda > debug/test_task3_based_task5.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python test.py --trained_task task_5 --testing_task task_4 --trained_model checkpoints/task_5 --cuda > debug/test_task4_based_task5.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python test.py --trained_task task_5 --testing_task task_5 --trained_model checkpoints/task_5 --cuda > debug/test_task5_based_task5.log 2>&1 &
