#!/bin/bash

python ./train.py --checkpoint=new  --batch-size 4 --num-epochs 5 --data-dir=./data/nerve-dataset  \
--num-classes 2 --optimizer 'sgd'