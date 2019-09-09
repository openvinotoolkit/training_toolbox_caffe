
# TTCF: Training Toolbox for Caffe

[![Build Status](http://134.191.240.124/buildStatus/icon?job=caffe-toolbox/develop/trigger)](http://134.191.240.124/job/caffe-toolbox/job/develop/job/trigger/)

This is a [BVLC Caffe](https://github.com/BVLC/caffe) fork that is intended for deployment multiple SSD-based detection models. It includes
- action detection and action recognition models for smart classroom use-case, see [README_AD.md](README_AD.md),
- person detection for smart classroom use-case, see [README_PD.md](README_PD.md),
- face detection model, see [README_FD.md](README_FD.md),
- person-vehicle-bike crossroad detection model, see [README_CR.md](README_CR.md),
- age & gender recognition model, see [README_AG.md](README_AG.md).

Please find original readme file [here](README_BVLC.md).


## Models
* [Action recognition](./README_AD.md)
* [Age & gender recognition](./README_AG.md)
* [Face detection](./README_FD.md)
* [Person detection](./README_PD.md)
* [Person-vehicle-bike crossroad detection](./README_CR.md)


## Install requirements
1. [Install Docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/)

**WARNING** Always examine scripts downloaded from the internet before running them locally.
```Shell
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

2. (optional) [Install nvidia-docker plugin](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0))
```Shell
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo pkill -SIGHUP dockerd
```

3. (optional) [Configure proxy settings](https://docs.docker.com/config/daemon/systemd/#httphttps-proxy)
Create a file `/etc/systemd/system/docker.service.d/proxy.conf` that adds the proxy environment variables:
```
[Service]
Environment="HTTP_PROXY=http://proxy.example.com:80/"
Environment="HTTPS_PROXY=https://proxy.example.com:443/"
```

Flush changes and restart Docker daemon
```
sudo systemctl daemon-reload
sudo systemctl restart docker
```


4. [Manage Docker as a non-root user]( https://docs.docker.com/engine/installation/linux/linux-postinstall/#manage-docker-as-a-non-root-user)
```Shell
sudo groupadd docker
sudo usermod -aG docker $USER
# Log out and log back in so that your group membership is re-evaluated.
```

5. (optional) Verify that nvidia-docker is installed correctly
```Shell
CUDA_VERSION=$(grep -oP '(?<=CUDA Version )(\d+)' /usr/local/cuda/version.txt)
nvidia-docker run --rm nvidia/cuda:${CUDA_VERSION}.0-cudnn7-devel-ubuntu16.04 nvidia-smi
```

## Build instructions
1. Get the code. We will call the directory that you cloned Caffe into `$CAFFE_ROOT`
```Shell
git clone https://github.com/opencv/training_toolbox_caffe.git caffe
```

2. [Download openvino package](https://software.intel.com/en-us/openvino-toolkit) to root directory of the repository

3. Build docker image
```Shell
./build_docker_image.sh gpu
```

## Run Docker interactive session
```
NV_GPU=0 nvidia-docker run --rm --name ttcf -it --user=$(id -u):$(id -g) -v <host_path>:<container_path> ttcf:gpu bash
```

**NOTE** To run in CPU mode
```
./build_docker_image.sh cpu
docker run --rm --name ttcf -it --user=$(id -u):$(id -g) -v <host_path>:<container_path> ttcf:cpu bash
```
And add to all scripts `--gpu -1 --image tccf:cpu` arguments.


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
