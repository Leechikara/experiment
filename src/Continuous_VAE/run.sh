#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 nohup python run.py > debug/run4.log 2>&1 &