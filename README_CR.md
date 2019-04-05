# Crossroad scenario.

## Data preparation

The training procedure can be done using data in LMDB format. Just as an example of usage there is a small dataset provided. At first, you need to create LMDB files from it. The original annotations are stored in <path_to_dataset>/annotation_train_cvt.json and <path_to_dataset>/annotation_val_cvt.json files.


### Create LMDB files

To create LMDB files go to the '$CAFFE_ROOT/python/lmdb_utils/' directory and run the following scripts:

1. Run docker in interactive sesion with mounted directory with WIDER dataset
```
nvidia-docker --rm -it -v <path_to_dataset>:/workspace tccf bash
```

2. Convert original annotation to Pascal VOC format for training subset. This way the annotation makes compatible with Caffe SSD tools, required for data LMDB generation.
```
python3 $CAFFE_ROOT/python/lmdb_utils/convert_to_voc_format.py <path_to_dataset>/annotation_train_cvt.json <path_to_dataset>/annotation_train_voc <path_to_dataset>/crossroad_100_train.txt
 ```
3. Set data_root_dir to <path_to_dataset> in $CAFFE_ROOT/python/lmdb_utils/create_cr_lmdb.sh script and run bash script to create LMDB:
```
nano $CAFFE_ROOT/python/lmdb_utils/create_cr_lmdb.sh
$CAFFE_ROOT/python/lmdb_utils/create_cr_lmdb.sh
 ```
4. Close docker session by 'alt+D' and check that you have lmdb files in <path_to_dataset>/lmdb.

5. Go to `models/crossroad` and set data_param field of data layer to `<path_to_dataset>/lmdb` in `train.prototxt` file.


###

### Person-vehicle-bike crossroad detection training
On next stage we should train the Person-vehicle-bike crossroad (four class) detection model. To do this follow next steps:

```Shell
cd ./models
python train.py --model crossroad \                                        # name of model
                --weights person-vehicle-bike-crossroad-0078.caffemodel \  # initialize weights from 'init_weights' directory
                --data_dir <PATH_TO_DATA> \                                # path to directory with dataset
                --work_dir<WORK_DIR> \                                     # directory to collect file from training process
                --gpu <GPU_ID>
```


### Person-vehicle-bike crossroad detection model evaluation
To evaluate the quality of trained Person-vehicle-bike crossroad detection model on your test data you can use provided scripts.

```Shell
python evaluate.py --type cr \
    --dir <EXPERIMENT_DIR> \
    --data_dir <DATA_DIR> \
    --annotaion <ANNOTATION_FILE> \
    --iter <ITERATION_NUM>
```

### Export to IR format

```Shell
python mo_convert.py --name crossroad \
    --dir <DATA_DIR>/crossroad/<EXPERIMENT_NUM> \
    --iter <ITERATION_NUM> \
    --data_type FP32
```

###  demo
You can use [this demo](https://github.com/opencv/open_model_zoo/tree/master/demos/crossroad_camera_demo) to view how resulting model performs.
