# Person detection

## Data preparation
Prepare dataset follow [instruction](./README_DATA.md)

**Note 1**: To significantly speedup training you can initialize your model from our distributed `.caffemodel` snapshots:
 * `$REPO_ROOT/models/init_weights/person_detection_0022.caffemodel` - for training Person Detection model


### Person Detection training
On first stage you should train the SSD-based person (two class) detector. To do this you should run single-GPU (python layers does not allow to run on multiple GPUs) training procedure (specify `GPU_ID`):


```Shell
cd ./models
python3 train.py --model person_detection \                   # name of model
                --weights person_detection_0022.caffemodel \  # initialize weights from 'init_weights' directory
                --data_dir <PATH_TO_DATA> \                   # path to directory with dataset
                --work_dir <WORK_DIR>                         # directory to collect file from training process
```

If it's needed the model evaluation can be performed by default pipeline in the original SSD [repository](https://github.com/weiliu89/caffe/tree/ssd). Moreover the training process of PD model can be carried out using SSD-original environment without any changes and after this the weights of trained model can be used as an initialization point on next [stage](#action-recognition-training).

Note: to get more accurate model it's recommended to use pre-training of backbone on default classification or detection datasets.


### Export to IR format

```Shell
cd ./models
python3 mo_convert.py --name face_detection \
    --dir <WORK_DIR>/person_detection/<EXPERIMENT_NUM> \
    --iter <INTERATION> \
    --data_type FP32
```
