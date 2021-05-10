#!/bin/bash

python ./train.py  --checkpoint=load  --checkpoint-file ./unet_best.params --data-dir=./data/nerve-dataset  \
--num-classes 2 --batch-size 1 --num-epochs 120 \
--optimizer 'adam' --learning-rate 0.001 --lr-decay 0.1 --lr-decay-epoch='40, 80, 100'