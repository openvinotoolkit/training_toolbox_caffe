# Smart classroom scenario.
This [BVLC Caffe](https://github.com/BVLC/caffe) fork contains code for deployment of action detection and action recognition models for smart classroom use-case. You can define own list of possible actions (see annotation file [format]() and steps for model training to change the list of actions) but this repository shows example for 3 action classes: standing, sitting and raising hand.


## Data preparation
Prepare dataset follow [instruction](./README_DATA.md)


## Model training
The train procedure for action detection&recognition model consists of two consistent stages:
 1. [Person Detection training (PD)](./README_PD.md)
 2. [Action Recognition training (AR)](#action-recognition-training)
 3. [Action Recognition model evaluation](#action-recognition-model-evaluation)
 4. [Conversion to MO-compatible format](#conversion-to-mo-compatible-format)


**Note 1**: To significantly speedup training you can initialize your model from our distributed `.caffemodel` snapshots:
 * `$REPO_ROOT/models/init_weights/action_detection_0005.caffemodel`

**Note 2**: if you want to change the list of supported actions follow next steps:

1. Use template `person_detection_action_recognition_N_classes` to generate file to train model for N action
```Shell
cd ./models/templates/person_detection_action_recognition_N_classes
./generate.py -n <NUMBER_OF_ACTION> --model_name person_detection_action_recognition_2_classes
```

2. Change fields `class_names_map` and `valid_class_names` in `data_config.json`
```json
"class_names_map": {
  "class_label_0": 0,
  "class_label_1": 1,
  "__undefined__": 2
},
"valid_class_names": ["class_label_0", "class_label_1"]
```


### (optional) Prepare init weights from PD model
1. Run docker in interactive sesion with mounted directory with WIDER dataset
```
nvidia-docker --rm -it -v <path_to_folder_with_weights>:/workspace tccf bash
```

2. To initialize AR model part copy weights from twin PD branch:
```
python2 $CAFFE_ROOT/python/rename_layers.py -i pd_weights_path.cafemodel -o ar_init_weights_path.cafemodel -p "cl/"
```
where `pd_weights_path.cafemodel` - weights of trained PD model (see [previous](#person-detection-training) section) and `ar_init_weights_path.cafemodel` - output path to init weights for AR model.


3. Move `pd_weights_path.cafemodel` and `ar_init_weights_path.cafemodel` to `init_weights` directory

**NOTE**: `train.py` should be run with `-w "pd_weights_path.cafemodel,ar_init_weights_path.cafemodel"`


### Action Recognition training
On next stage we should train the Action Recognition (AR) model which reuses detections from Person Detector (PD) model part and assigns action label for each of them. To do this follow next steps:

```Shell
cd ./models
python train.py --model person_detection_action_recognition \ # name of model
                --weights action_detection_0005.caffemodel \  # initialize weights from 'init_weights' directory
                --data_dir <PATH_TO_DATA> \                   # path to directory with dataset
                --work_dir <WORK_DIR> \                       # directory to collect file from training process
                --gpu <GPU_ID>
```


### Action Recognition model evaluation
To evaluate the quality of trained Action Recognition model on your test data you can use provided script. To do this you need the file with testing tasks in the same format as for training stage (see [this](#train-tasks-file-format) section). The model can be evaluated in two modes:

1. Frame independent evaluation:
```Shell
python evaluate.py --type ad \
    --dir <EXPERIMENT_DIR> \
    --data_dir <DATA_DIR> \
    --annotaion test_tasks.txt \
    --iter <ITERATION_NUM>
```

2. Event-based evaluation:
```Shell
python evaluate.py --type ad_event \
    --dir <EXPERIMENT_DIR> \
    --data_dir <DATA_DIR> \
    --annotaion test_tasks.txt \
    --iter <ITERATION_NUM>
```

### Export to IR format

```Shell
python mo_convert.py --name action_recognition --type ad \
    --dir <EXPERIMENT_DIR> \
    --iter <ITERATION_NUM> \
    --data_type FP32
```
