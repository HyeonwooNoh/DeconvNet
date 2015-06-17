## DeconvNet: *Learning Deconvolution Network for Semantic Segmentation*

Created by Hyeonwoo Noh, Seunghoon Hong and Bohyung Han at POSTECH

Acknowledgements: Thanks to Yangqing Jia and the BVLC team for creating Caffe.

### Introduction

DeconvNet is state-of-the-art semantic segmentation system that combines bottom-up region proposals with multi-layer decovolution network.

Detailed description of the system will be provided by our technical report [arXiv tech report] http://arxiv.org/abs/1505.04366

### Citation

If you're using this code in a publication, please cite our papers.

Hyeonwoo Noh, Seunghoon Hong, Bohyung Han.

Learning Deconvolution Network for Semantic Segmentation

arXiv:1505.04366, 2015.

    @article{noh2015learning,
      title={Learning Deconvolution Network for Semantic Segmentation},
      author={Noh, Hyeonwoo and Hong, Seunghoon and Han, Bohyung},
      journal={arXiv preprint arXiv:1505.04366},
      year={2015}
    }


### Licence

This software is being made available for research purpose only.

check LICENSE file for details.

### System Requirements

This software is tested on Ubuntu 14.04 LTS (64bit).

**Prerequisites** 
  0. MATLAB (tested with 2014b on 64-bit Linux)
  0. prerequisites for caffe(http://caffe.berkeleyvision.org/installation.html#prequequisites)

### Installing DeconvNet

** By running "setup.sh" you can download all the necessary file for training and inference include: **
  0. caffe: you need modified version of caffe which support DeconvNet - https://github.com/HyeonwooNoh/caffe.git
  0. data: data used for training stage 1 and 2
  0. model: caffemodel of trained DeconvNet and other caffemodels required for training

### Training DeconvNet

Training scripts are included in "./training/" directory

You can simply run following scripts in order to train DeconvNet
  0. 001\_start\_train.sh : script for first stage training
  0. 002\_start\_train.sh : script for second stage training
  0. 003\_start\_make\_bn\_layer\_testable : script converting trained DeconvNet with bn layer to inference mode








 
