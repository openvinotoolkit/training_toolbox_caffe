
# TTCF: Training Toolbox for Caffe

[![Build Status](http://134.191.240.124/buildStatus/icon?job=caffe-toolbox/develop/trigger)](http://134.191.240.124/job/caffe-toolbox/job/develop/job/trigger/)

This is a [BVLC Caffe](https://github.com/BVLC/caffe) fork that is intended for deployment multiple SSD-based detection models. It includes
- action detection and action recognition models for smart classroom use-case, see [README_AD.md](README_AD.md),
- face detection model, see [README_FD.md](README_FD.md).

Please find original readme file [here](README_BVLC.md).

If you want to make a contribution please follow [the guideline](CONTRIBUTING.md).


## Build instructions
1. Get the code. We will call the directory that you cloned Caffe into `$CAFFE_ROOT`
  ```Shell
  git clone https://github.com/opencv/training_toolbox_caffe.git caffe
  cd caffe
  git checkout develop
  ```
2. Build the code. Please follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.
  ```Shell
  sudo pip install -r $CAFFE_ROOT/python/requirements.txt
  mkdir build && cd build
  cmake ..
  make -j8
  export PYTHONPATH=$PYTHONPATH:$CAFFE_ROOT/python
  # (Optional)
  make runtest -j8
  make pytest
  ```

## License and Citation

### Original Caffe
Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BAIR/BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }

### SSD: Single Shot MultiBox Detector
Please cite SSD in your publications if it helps your research:

    @inproceedings{liu2016ssd,
      title = {{SSD}: Single Shot MultiBox Detector},
      author = {Liu, Wei and Anguelov, Dragomir and Erhan, Dumitru and Szegedy, Christian and Reed, Scott and Fu, Cheng-Yang and Berg, Alexander C.},
      booktitle = {ECCV},
      year = {2016}
    }

### AM-Softmax
If you find **AM-Softmax** useful in your research, please consider to cite:

    @article{Wang_2018_amsoftmax,
      title = {Additive Margin Softmax for Face Verification},
      author = {Wang, Feng and Liu, Weiyang and Liu, Haijun and Cheng, Jian},
      journal = {arXiv preprint arXiv:1801.05599},
      year = {2018}
    }

### WIDERFace dataset

    @inproceedings{yang2016wider,
      Author = {Yang, Shuo and Luo, Ping and Loy, Chen Change and Tang, Xiaoou},
      Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      Title = {WIDER FACE: A Face Detection Benchmark},
      Year = {2016}
    }
