ARG BASE_IMAGE=nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        nano \
        cmake \
        git \
        wget \
        tar \
        bzip2 \
        unzip \
        gzip \
        autoconf \
        automake \
        libtool \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libsnappy-dev \
        python3-dev \
        python3-pip \
        libgtk2.0-0 \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        libxvidcore-dev \
        libx264-dev \
        cpio \
        base-files \
        lsb-release \
        lsb-base \
        libgstreamer-plugins-base1.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
RUN wget  https://github.com/google/protobuf/releases/download/v3.1.0/protobuf-cpp-3.1.0.tar.gz && \
    tar xvzf protobuf-cpp-3.1.0.tar.gz && \
    cd /tmp/protobuf-3.1.0 && \
    ./autogen.sh && \
    ./configure --prefix=/usr && \
    make -j 4 && \
    make check && \
    make install && \
    rm -rf /tmp/protobuf*

ARG OPENVINO_VERSION=2018.5.455
COPY l_openvino_toolkit_p_${OPENVINO_VERSION}.tgz /tmp/l_openvino_toolkit_p_${OPENVINO_VERSION}.tgz
WORKDIR /tmp
RUN tar -xvzf l_openvino_toolkit_p_${OPENVINO_VERSION}.tgz && \
    cd l_openvino_toolkit_p_${OPENVINO_VERSION} && \
    sed -i -e 's/ACCEPT_EULA=decline/ACCEPT_EULA=accept/g' silent.cfg && \
    sh install.sh -s silent.cfg && \
    rm -rf /tmp/l_openvino* && \
    echo "source /opt/intel/openvino/bin/setupvars.sh" >> /etc/bash.bashrc

ENV CAFFE_ROOT=/opt/caffe

COPY caffe /opt/caffe
WORKDIR $CAFFE_ROOT

RUN pip3 install --upgrade pip==9.0.3 && \
    pip3 install --upgrade setuptools wheel && \
    pip3 install -r python/requirements.txt && \
    grep -v protobuf /opt/intel/openvino/deployment_tools/model_optimizer/requirements_caffe.txt | xargs -n 1 pip3 install

RUN mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
      -Dpython_version=3 \
      -Wno-dev \
      -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
      -DCUDA_ARCH_BIN="30 35 50 52 60 61 70" \
      -DCUDA_ARCH_PTX="" \
      -DCUDA_ARCH_NAME="Manual" \
      -DCMAKE_CXX_FLAGS="-std=c++11" \
      -DOpenCV_DIR=/opt/intel/openvino/opencv/cmake \
      .. && \
    make -j 4 && make pycaffe

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig &> /dev/null

RUN echo 'export PS1="\w\$ "' >> /etc/bash.bashrc

WORKDIR /workspace
