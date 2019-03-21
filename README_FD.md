# Face detection.

## Data preparation

The training procedure can be done using data in LMDB format. To launch training or evaluation at the WiderFace dataset, download it from [the source](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/), extract images and annotations into <path_to_widerface_root_folder> folder and use the provided scripts to convert original annotations to LMDB format.

### Create LMDB files

To create LMDB files got to the '$CAFFE_ROOT/python/lmdb_utils/' directory and run the following scripts:

1. Run docker in interactive sesion with mounted directory with WIDER dataset
```
nvidia-docker --rm -it -v <path_to_widerface_root_folder>:<path_to_widerface_root_folder> tccf bash
```

2.  Convert original annotation to xml format for both train and val subsets:
```
python3 $CAFFE_ROOT/python/lmdb_utils/wider_to_xml.py <path_to_widerface_root_folder> <path_to_widerface_root_folder>/WIDER_train/images/ <path_to_widerface_root_folder>/wider_face_split/wider_face_train_bbx_gt.txt train
python3 $CAFFE_ROOT/python/lmdb_utils/wider_to_xml.py <path_to_widerface_root_folder> <path_to_widerface_root_folder>/WIDER_train/images/ <path_to_widerface_root_folder>/wider_face_split/wider_face_val_bbx_gt.txt val
```

2. Convert xml annotations to set of xml files per image:
```
python3 $CAFFE_ROOT/python/lmdb_utils/xml_to_ssd.py --ssd_path <path_to_widerface_root_folder> --xml_path_train <path_to_widerface_root_folder>/wider_train.xml --xml_path_val <path_to_widerface_root_folder>/wider_val.xml
 ```

3. Set data_root_dir to <path_to_widerface_root_folder> in $CAFFE_ROOT/python/lmdb_utils/create_data.sh script and run bash script to create LMDB:
```
nano ./$CAFFE_ROOT/python/lmdb_utils/create_data.sh
./$CAFFE_ROOT/python/lmdb_utils/create_data.sh
 ```

4. Close docker session by 'alt+D' and check that you have lmdb files in <path_to_widerface_root_folder>.

5. Go to `models/face_detection` and replace in `test.prototxt` and `train.protoxt` files `<path_to_widerface>` to your path.


###

### Action Recognition training
On next stage we should train the Action Recognition (AR) model which reuses detections from Person Detector (PD) model part and assigns action label for each of them. To do this follow next steps:

```Shell
cd ./models
python train.py --model face_detection \                           # name of model
                --weights face-detection-retail-0044.caffemodel \  # initialize weights from 'init_weights' directory
                --data_dir <PATH_TO_DATA> \                        # path to directory with dataset
                --work_dir<WORK_DIR> \                             # directory to collect file from training process
                --gpu <GPU_ID>
```


### Face Detection model evaluation
To evaluate the quality of trained Face Detection model on your test data you can use provided scripts.

```Shell
python evaluate.py --type fd \
    --dir <EXPERIMENT_DIR> \
    --data_dir <DATA_DIR> \
    --annotaion <ANNOTATION_FILE> \
    --iter <ITERATION_NUM>
```

### Export to IR format

```Shell
python mo_convert.py --name face_detection \
    --dir <DATA_DIR>/face_detection/<EXPERIMENT_NUM> \
    --iter <ITERATION_NUM> \
    --data_type FP32
```

### Face Detection demo
You can use [this demo](https://github.com/opencv/open_model_zoo/tree/master/demos/interactive_face_detection_demo) to view how resulting model performs.
