# U-NET reference implementation in MXNET

Reference implementation of [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

## Dataset

There are multiple datasets provided for testing the implementation.
They have been downloaded from sources bellow and extracted into `dataset` folder.

* [ISBI Challenge: Segmentation of neuronal structures in EM stacks](http://brainiac2.mit.edu/isbi_challenge/)


## Usage


### Training

```sh
python ./segmenter.py --checkpoint-file ./unet_best.params
```

### Restarting training

```sh
python ./segmenter.py --checkpoint=load --checkpoint-file ./unet_best.params
```

### Evaluating model

```sh
python ./evaluate.py --image=./input.png ----network-param./unet_best.params
```


## U-Net network features

* Resnet Block
  * Added Residual/Skip connection (ResBlock) as the original paper did not include them   
* Batch Norm / Layer Norm
  * Added normalization layer
* Dropout
  * Added support for optional dropout layer
* Conv2DTranspose / UpSample
  * Added support to switche between `Conv2DTranspose` and `UpSampling` http://distill.pub/2016/deconv-checkerboard/
  



## Dependencies 

* MXNET >= 1.7 

```sh
python -m pip install git+https://github.com/aleju/imgaug
python -m pip install mxboard
```