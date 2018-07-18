#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 nohup python run.py --cuda > debug/run.log 2>&1 &