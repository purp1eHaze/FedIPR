#!/bin/bash

python main_fedIPR.py --num_back 10  --num_trigger 10 --num_sign 10 --num_bit 40 --num_users 10 --dataset cifar10 --model_name alexnet --epochs 200 --gpu 0 --local_ep 2 --iid 


