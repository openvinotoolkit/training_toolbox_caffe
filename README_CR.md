# Crossroad scenario.

## Data preparation

As an example of usage please download a small dataset from [here](https://download.01.org/opencv/openvino_training_extensions/datasets/crossroad/crossroad_85.tar.gz). To run training, you firstly need to create LMDB files. The annotations should be stored in <DATA_DIR>/annotation_train_cvt.json and <DATA_DIR>/annotation_val_cvt.json files.


### Create LMDB files

To create LMDB files go to the '$CAFFE_ROOT/python/lmdb_utils/' directory and run the following scripts:

1. Run docker in interactive sesion with mounted directory with WIDER dataset
```Shell
nvidia-docker run --rm -it --user=$(id -u) -v <DATA_DIR>:/data ttcf bash
```

2. Convert original annotation to Pascal VOC format for training subset. This way the annotation makes compatible with Caffe SSD tools, required for data LMDB generation.
```Shell
python3 $CAFFE_ROOT/python/lmdb_utils/convert_to_voc_format.py /data/annotation_train_cvt.json /data/train.txt
```
3. Run bash script to create LMDB:
```Shell
bash $CAFFE_ROOT/python/lmdb_utils/create_cr_lmdb.sh
```
4. Close docker session by 'alt+D' and check that you have lmdb files in <DATA_DIR>/lmdb.


###

### Person-vehicle-bike crossroad detection training
On next stage we should train the Person-vehicle-bike crossroad (four class) detection model. To do this follow next steps:

```Shell
cd ./models
python train.py --model crossroad \                                        # name of model
                --weights person-vehicle-bike-crossroad-0078.caffemodel \  # initialize weights from 'init_weights' directory
                --data_dir <DATA_DIR> \                                    # path to directory with dataset
                --work_dir<WORK_DIR> \                                     # directory to collect file from training process
                --gpu <GPU_ID>
```


### Person-vehicle-bike crossroad detection model evaluation
To evaluate the quality of trained Person-vehicle-bike crossroad detection model on your test data you can use provided scripts.

```Shell
python evaluate.py --type cr \
    --dir <WORK_DIR>/crossroad/<EXPERIMENT_NUM> \
    --data_dir <DATA_DIR> \
    --annotation annotation_val_cvt.json \
    --iter <ITERATION_NUM>
```

### Export to IR format

```Shell
python mo_convert.py --type cr \
    --name crossroad \
    --dir <WORK_DIR>/crossroad/<EXPERIMENT_NUM> \
    --iter <ITERATION_NUM> \
    --data_type FP32
```

###  demo
You can use [this demo](https://github.com/opencv/open_model_zoo/tree/master/demos/crossroad_camera_demo) to view how resulting model performs.
