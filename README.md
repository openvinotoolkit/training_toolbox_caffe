This is a [BVLC Caffe](https://github.com/BVLC/caffe) fork that is intended for deployment of models that are benchmarked and/or developed inside ICV. That means that checking out master branch of the project should be enough to run any Caffe model that is mentioned in ICV benchmark report and/or delivered by ICV as a part of some capability.

Windows x64 build (tested with MSVC 2015)
1. install Boost 1.65.1, precompiled msvc 14.0 x64 install package can be downloaded by [the link below](https://sourceforge.net/projects/boost/files/boost-binaries/1.65.1/boost_1_65_1-msvc-14.0-64.exe/download)
2. install HDF5 1.8.19, install package can be downloaded by [the link below](https://support.hdfgroup.org/ftp/HDF5/current18/bin/windows/hdf5-1.8.19-Std-win7_64-vs2015.zip)
3. install Intel Computer Vision SDK 1.0 R3 (needed for OpenCV availability)
4. install Python if python layer needed. Tested with Anaconda 5.0.1 Python 2.7,
   install package can be downloaded by [the link below](https://repo.continuum.io/archive/Anaconda2-5.0.1-Windows-x86_64.exe)
   To successfully build the python interface you need to add the following conda channels:
     ```
     conda config --add channels conda-forge
     conda config --add channels willyd
     ```
     and install the following packages:
     ```
     conda install --yes cmake ninja numpy scipy protobuf==3.1.0 six scikit-image pyyaml pydotplus graphviz
     ```
   If Python is installed the default is to build the python interface and python layers.
   If you wish to disable the python layers or the python build use the CMake options `-DBUILD_python_layer=0` and
   `-DBUILD_python=0` respectively. In order to use the python interface you need to either add the
   `C:\Projects\caffe\python` folder to your python path of copy the `C:\Projects\caffe\python\caffe` folder to your
   `site_packages` folder.
5. Configuration and build caffe on Windows tested with the following CMake options:
   `-DOpenCV_DIR=<path to opencv within Intel CV SDK>`
   `-DHDF5_DIR=<path to installation folder of HDF5 package>`
   `-DPYTHON_EXECUTABLE=<full path name of python.exe within Anaconda2 installation folder>`
   `-DCPU_ONLY=ON`
   `-DBLAS=MKL`
   `-DUSE_CUDNN=OFF`
   `-DUSE_NCCL=OFF`
   `-DUSE_OPENCV=ON`
   `-DUSE_LEVELDB=OFF`
   `-DUSE_LMDB=OFF`
   `-DBUILD_python=ON`
   `-DBUILD_python_layer=ON`
   `-DBUILD_matlab=OFF`
   `-DBUILD_docs=OFF`
6. Note, you will need to add path to GFlags DLL, found at `<BUILD_FOLDER>/external/gflags-install/bin` folder in order
   to run application linked with caffe.dll

Please find original readme file [here](README_BVLC.md).

If you want to make a contribution please follow [the guideline](CONTRIBUTING.md).
