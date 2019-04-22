# Face detection.

## Data preparation

The training procedure can be done using data in LMDB format. To launch training or evaluation at the WiderFace dataset, download it from [the source](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/), extract images and annotations into <DATA_DIR> folder and use the provided scripts to convert original annotations to LMDB format.

### Create LMDB files

To create LMDB files go to the '$CAFFE_ROOT/python/lmdb_utils/' directory and run the following scripts:

1. Run docker in interactive sesion with mounted directory with WIDER dataset
```Shell
nvidia-docker run --rm -it --user=$(id -u) -v <DATA_DIR>:/data ttcf bash
```

2.  Convert original annotation to xml format for both train and val subsets:
```Shell
python3 $CAFFE_ROOT/python/lmdb_utils/wider_to_xml.py /data /data/WIDER_train/images/ /data/wider_face_split/wider_face_train_bbx_gt.txt train
python3 $CAFFE_ROOT/python/lmdb_utils/wider_to_xml.py /data /data/WIDER_val/images/ /data/wider_face_split/wider_face_val_bbx_gt.txt val
```

3. Convert xml annotations to set of xml files per image:
```Shell
python3 $CAFFE_ROOT/python/lmdb_utils/xml_to_ssd.py --ssd_path /data --xml_path_train /data/wider_train.xml --xml_path_val /data/wider_val.xml
```

4. Run bash script to create LMDB:
```Shell
bash $CAFFE_ROOT/python/lmdb_utils/create_wider_lmdb.sh
```

5. Close docker session by 'alt+D' and check that you have lmdb files in <DATA_DIR>.


###

### Face detection training
On next stage we should train the Face Detection model. To do this follow next steps:

```Shell
cd ./models
python train.py --model face_detection \                           # name of model
                --weights face-detection-retail-0044.caffemodel \  # initialize weights from 'init_weights' directory
                --data_dir <DATA_DIR> \                            # path to directory with dataset
                --work_dir <WORK_DIR> \                            # directory to collect file from training process
                --gpu <GPU_ID>
```


### Face Detection model evaluation
To evaluate the quality of trained Face Detection model on your test data you can use provided scripts.

```Shell
python evaluate.py --type fd \
    --dir <WORK_DIR>/face_detection/<EXPERIMENT_NUM> \
    --data_dir <DATA_DIR> \
    --annotation wider_val.xml \
    --iter <ITERATION_NUM>
```

### Export to IR format

```Shell
python mo_convert.py --name face_detection \
    --dir <WORK_DIR>/face_detection/<EXPERIMENT_NUM> \
    --iter <ITERATION_NUM> \
    --data_type FP32
```

### Face Detection demo
You can use [this demo](https://github.com/opencv/open_model_zoo/tree/master/demos/interactive_face_detection_demo) to view how resulting model performs.
