# Age & gender recognition.

## Data preparation

The training procedure can be done using data in HDF5 format. Please, prepare images with faces and put it into some
<DATA_DIR> folder. Then create a special file (<DATA_FILE>) for 'train', 'val' and 'test' phase containing annotations
with the following structure:
```
image_1_relative_path <gender> <age/100>
...
image_n_relative_path <gender> <age/100>
```
The example images with a corresponding data file can be found in ./data directory and used in evaluation script.

Once you have images and a data file, use the provided script to create database in HDF5 format.

### Create HDF5 files
1. Run docker in interactive sesion with mounted directory of your data
```Shell
nvidia-docker run --rm -it --user=$(id -u) -v <DATA_DIR>:/data ttcf bash
```

2. Run the script to convert data to hdf5 format
 ```Shell
python3 $CAFFE_ROOT/python/gen_hdf5_data.py /data/<DATA_TRAIN_FILE> images_db_train
python3 $CAFFE_ROOT/python/gen_hdf5_data.py /data/<DATA_VAL_FILE> images_db_val
python3 $CAFFE_ROOT/python/gen_hdf5_data.py /data/<DATA_TEST_FILE> images_db_test
```

3. Close docker session by 'alt+D' and check that you have images_db_<subset>.hd5 and images_db_<subset>_list.txt files in <DATA_DIR>.


## Model training and evaluation

### Age-gender recognition model training
On next stage we should train the Age-gender recognition model. To do this follow next steps:

```Shell
cd ./models
python train.py --model age_gender \                                       # name of model
                --weights age-gender-recognition-retail-0013.caffemodel \  # initialize weights from 'init_weights' directory
                --data_dir <DATA_DIR> \                                    # path to directory with dataset
                --work_dir <WORK_DIR> \                                    # directory to collect file from training process
                --gpu <GPU_ID>
```


### Age-gender recognition model evaluation
To evaluate the quality of trained Age-gender recognition model on your test data you can use provided scripts.

```Shell
python evaluate.py --type ag \
    --dir <WORK_DIR>/age_gender/<EXPERIMENT_NUM> \
    --data_dir <DATA_DIR> \
    --annotation <DATA_FILE> \
    --iter <ITERATION_NUM>
```

### Export to IR format

```Shell
python mo_convert.py --name age_gender --type ag \
    --dir <WORK_DIR>/age_gender/<EXPERIMENT_NUM> \
    --iter <ITERATION_NUM> \
    --data_type FP32
```

### Age & gender recognition demo
You can use [this demo](https://github.com/opencv/open_model_zoo/tree/master/demos/interactive_face_detection_demo) to view how resulting model performs.
