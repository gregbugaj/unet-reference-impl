#!/bin/bash

python ./train.py  --gpu-id 0 --checkpoint=new  --data-dir=./data/nerve-dataset  \
--num-classes 2 --batch-size 1 --num-epochs 40 \
--optimizer 'adam' --learning-rate 1e-4 --lr-decay 0.1 --lr-decay-epoch='5, 10, 20, 40, 60, 80, 100'