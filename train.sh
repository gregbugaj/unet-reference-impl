#!/bin/bash

python ./train.py  --gpu-id 0 --checkpoint=new  --data-dir=./data/nerve-dataset  \
--num-classes 2 --batch-size 1 --num-epochs 80 \
--optimizer 'sgd' --learning-rate .001 --lr-decay 0.1 --lr-decay-epoch='40, 60, 80'