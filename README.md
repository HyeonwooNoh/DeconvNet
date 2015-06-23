## DeconvNet: Learning Deconvolution Network for Semantic Segmentation

Created by [Hyeonwoo Noh](http://cvlab.postech.ac.kr/~hyeonwoonoh/), [Seunghoon Hong](http://cvlab.postech.ac.kr/~maga33/) and [Bohyung Han](http://cvlab.postech.ac.kr/~bhhan/) at POSTECH

Acknowledgements: Thanks to Yangqing Jia and the BVLC team for creating Caffe.

### Introduction

DeconvNet is state-of-the-art semantic segmentation system that combines bottom-up region proposals with multi-layer decovolution network.

Detailed description of the system will be provided by our technical report [arXiv tech report] http://arxiv.org/abs/1505.04366

### Citation

If you're using this code in a publication, please cite our papers.

    @article{noh2015learning,
      title={Learning Deconvolution Network for Semantic Segmentation},
      author={Noh, Hyeonwoo and Hong, Seunghoon and Han, Bohyung},
      journal={arXiv preprint arXiv:1505.04366},
      year={2015}
    }

### Pre-trained Model

If you need model definition and pre-trained model only, you can download them from following location:
  0. caffe for DeconvNet: https://github.com/HyeonwooNoh/caffe
  0. DeconvNet model definition: http://cvlab.postech.ac.kr/research/deconvnet/model/DeconvNet/DeconvNet_inference_deploy.prototxt
  0. Pre-trained DeconvNet weight: http://cvlab.postech.ac.kr/research/deconvnet/model/DeconvNet/DeconvNet_trainval_inference.caffemodel

### Licence

This software is being made available for research purpose only.
Check LICENSE file for details.

### System Requirements

This software is tested on Ubuntu 14.04 LTS (64bit).

**Prerequisites** 
  0. MATLAB (tested with 2014b on 64-bit Linux)
  0. prerequisites for caffe(http://caffe.berkeleyvision.org/installation.html#prequequisites)

### Installing DeconvNet

**By running "setup.sh" you can download all the necessary file for training and inference include:**
  0. caffe: you need modified version of caffe which support DeconvNet - https://github.com/HyeonwooNoh/caffe.git
  0. data: data used for training stage 1 and 2
  0. model: caffemodel of trained DeconvNet and other caffemodels required for training

### Training DeconvNet

Training scripts are included in *./training/* directory

**To train DeconvNet you can simply run following scripts in order:**
  0. 001\_start\_train.sh : script for first stage training
  0. 002\_start\_train.sh : script for second stage training
  0. 003\_start\_make\_bn\_layer\_testable : script converting trained DeconvNet with bn layer to inference mode

### Inference EDeconvNet+CRF

Run *run_demo.m* to reproduce EDeconvNet+CRF results on VOC2012 test data.

**This script will generated EDeconvNet+CRF results through following steps:**
  0. run FCN-8s and cache the score [cache\_FCN8s\_results.m]
  0. generate DeconvNet score and apply ensemble with FCN-8s score, post processing with densecrf [generate\_EDeconvNet\_CRF\_results.m]

*EDeconvNet+CRF obtains 72.5 mean I/U on PASCAL VOC 2012 Test*

**External dependencies [can be downloaded by running "setup.sh" script]**
  0. FCN-8s model and weight file [https://github.com/BVLC/caffe/wiki/Model-Zoo]
  0. densecrf with matlab wrapper [https://github.com/johannesu/meanfield-matlab.git]
  0. cached proposal bounding boxes extracted with edgebox object proposal [https://github.com/pdollar/edges] 






 
