# Face detection.

## Data preparation

The training procedure can be done using data in LMDB format. To launch training or evaluation at the WiderFace dataset, download it from [the source](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/), extract images and annotations into <path_to_widerface_root_folder> folder and use the provided scripts to convert original annotations to LMDB format.

### Create LMDB files

To create LMDB files got to the '$CAFFE_ROOT/python/lmdb_utils/' directory and run the following scripts:
 1. Convert original annotation to xml format for both train and val subsets:
```
PYTHONPATH=$PYTHONPATH:$CAFFE_ROOT/python python $CAFFE_ROOT/python/lmdb_utils/wider_to_xml.py <path_to_widerface_root_folder> <path_to_widerface_root_folder>/WIDER_train/images/ <path_to_widerface_root_folder>/wider_face_split/wider_face_train_bbx_gt.txt train
 ```
```
PYTHONPATH=$PYTHONPATH:$CAFFE_ROOT/python python $CAFFE_ROOT/python/lmdb_utils/wider_to_xml.py <path_to_widerface_root_folder> <path_to_widerface_root_folder>/WIDER_val/images/ <path_to_widerface_root_folder>/wider_face_split/wider_face_val_bbx_gt.txt val
 ```
 2. Convert xml annotations to set of xml files per image:
```
PYTHONPATH=$PYTHONPATH:$CAFFE_ROOT/python python $CAFFE_ROOT/python/lmdb_utils/xml_to_ssd.py --ssd_path <path_to_widerface_root_folder> --xml_path_train <path_to_widerface_root_folder>/wider_train.xml --xml_path_val <path_to_widerface_root_folder>/wider_val.xml
 ```
 3. Set data_root_dir to <path_to_widerface_root_folder> in $CAFFE_ROOT/python/lmdb_utils/create_data.sh script and run bash script to create LMDB:
```
./$CAFFE_ROOT/python/lmdb_utils/create_data.sh
 ```
 4. Find final LMDB files in <path_to_widerface_root_folder>/lmdb directory and set the correct paths to it in $CAFFE_ROOT/models/face_detection/train.prototxt and $CAFFE_ROOT/models/face_detection/test.prototxt:
    in data layer set 'source: <path_to_widerface_root_folder>/lmdb/wider_wider_<val/train>_lmdb'
    in detection_eval layer of test.prototxt set 'name_size_file: <path_to_widerface_root_folder>/list_wider_val_size.txt'

###

## Model training and evaluation
The flow consists of three stages:

 1. [Face Detection training (FD)](#face-detection-training)
 2. [Face Detection model evaluation](#face-detection-model-evaluation)
 3. [Export to IR format](#export-to-ir-format)

We suggest you to initialize your model from our distributed  `.caffemodel` snapshot:  `$CAFFE_ROOT/models/face_detection/face-detection-retail-0044.caffemodel` and continue with fine tuning.

### Face Detection training
Create a folder for resulting model:
```
mkdir snapshots
 ```
By default the resulting model snapshots will be stored in the ./snapshots folder and prefixed with 'fd'. You can change that it the $CAFFE_ROOT/models/face_detection/solver.prototxt file.

To train the SSD-based face (two class) detector you should run single-GPU (python layers does not allow to run on multiple GPUs) training procedure (specify `GPU_ID`):
```
PYTHONPATH=$PYTHONPATH:$CAFFE_ROOT/python $CAFFE_ROOT/build/tools/caffe train --gpu=GPU_ID --solver=$CAFFE_ROOT/models/face_detection/solver.prototxt --weights=$CAFFE_ROOT/models/face_detection/face-detection-retail-0044.caffemodel
 ```

### Face Detection model evaluation
To evaluate the quality of trained Face Detection model on your test data you can use provided scripts. 
 1. Get found detections and save them in the file detections.xml:
```
PYTHONPATH=$PYTHONPATH:$CAFFE_ROOT/python python $CAFFE_ROOT/python/get_detections.py --gt <path_to_widerface_root_folder>/wider_val.xml --def $CAFFE_ROOT/models/face_detection/deploy.prototxt --net $CAFFE_ROOT/snapshots/<path_to_resulting_model>.caffemodel --labels "['background','face']" --resize_to 300x300 --delay 1 --det detections.xml
 ```
 2. Evaluate found detections:
```
PYTHONPATH=$PYTHONPATH:$CAFFE_ROOT/python python $CAFFE_ROOT/python/eval_detections.py --gt <path_to_widerface_root_folder>/wider_val.xml --det detections.xml --objsize 16 1024 --imsize 1024 1024 --reasonable --mm --class_lbl face
 ```

### Export to IR format

 Run model optimizer:
```
python3 mo_caffe.py --input_model ./snapshots/<path_to_resulting_model>.caffemodel --input_proto ./models/face_detection/deploy.prototxt
 ```
You can use [this demo](https://github.com/opencv/open_model_zoo/tree/master/demos/interactive_face_detection_demo) to view how resulting model performs.

